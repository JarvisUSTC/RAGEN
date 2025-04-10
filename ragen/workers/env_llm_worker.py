from verl.workers.fsdp_workers import *
from verl import DataProto
import torch
import logging

logger = logging.getLogger(__name__)

class EnvironmentLLMWorker(Worker):
    """
    Worker for the environment LLM (patient responses in medical consultation).
    This worker handles generation of responses from the environment LLM.
    """
    
    def __init__(self, config: DictConfig, role: str = 'env_llm'):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        
        self.role = role
        self._is_offload_param = self.config.fsdp_config.get('param_offload', False)
        
        # Initialize model and tokenizer
        self._build_model_optimizer()
        
    def _build_model_optimizer(self):
        from verl.utils.model import print_model_size, update_model_config, get_generation_config
        from verl.utils.torch_dtypes import PrecisionType
        from transformers import AutoModelForCausalLM, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision, CPUOffload
        from torch import optim
        
        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_to_local(self.config.model.path)
        
        # Initialize tokenizer
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        self.processor = hf_processor(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        
        # Set torch dtype
        torch_dtype = self.config.fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)
            
        # Get model config
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        self.generation_config = get_generation_config(local_path, trust_remote_code=self.config.model.get('trust_remote_code', False))
        
        # Override model config
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(self.config.model.get('override_config', {}))
        update_model_config(model_config, override_config_kwargs=override_config_kwargs)
        
        # Initialize model
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings,
                                                       mesh=self.device_mesh)
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation='flash_attention_2',
                trust_remote_code=self.config.model.get('trust_remote_code', False)
            )
            
            # Apply Liger kernel if specified
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)
                
            self.model.to(torch_dtype)
            
        torch.distributed.barrier()
        
        if self.rank == 0:
            print_model_size(self.model)
            
        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)
        
        # Set up FSDP
        mixed_precision_config = self.config.fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32
            
        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)
        
        auto_wrap_policy = get_fsdp_wrap_policy(module=self.model, config=self.config.fsdp_config.get('wrap_policy', None))
        
        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)
        
        # Use CPUOffload to save memory
        cpu_offload = CPUOffload(offload_params=True)
        self.model_fsdp = FSDP(
            self.model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False
        )
        
        log_gpu_memory_usage('After Environment LLM FSDP init', logger=logger)
        
        # Get the original unwrapped module
        self.model = self.model_fsdp._fsdp_wrapped_module
        
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_responses(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(torch.cuda.current_device())
        
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.model_fsdp)
            
        # Support all hardwares
        prompts.batch = prompts.batch.to(torch.cuda.current_device())
        meta_info = {
            'eos_token_id': self.generation_config.eos_token_id if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id': self.generation_config.pad_token_id if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        
        with self.ulysses_sharding_manager:
            # After parameters sync, offload model to CPU
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.model_fsdp)
                
            log_gpu_memory_usage('Before environment LLM generation', logger=logger)
            
            prompts = self.ulysses_sharding_manager.preprocess_data(prompts)
            
            # Generate responses
            with torch.no_grad():
                outputs = self.model_fsdp.generate(
                    input_ids=prompts.batch['input_ids'],
                    attention_mask=prompts.batch['attention_mask'],
                    max_length=self.config.generation.get('max_length', 512),
                    temperature=self.config.generation.get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=meta_info['pad_token_id'],
                    eos_token_id=meta_info['eos_token_id'],
                )
                
            # Create output DataProto
            output = DataProto.from_dict({
                'responses': outputs
            })
            
            log_gpu_memory_usage('After environment LLM generation', logger=logger)
            
            output = self.ulysses_sharding_manager.postprocess_data(output)
            
        output = output.to('cpu')
        
        # Clear cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After environment LLM generation complete', logger=logger)
        
        return output 