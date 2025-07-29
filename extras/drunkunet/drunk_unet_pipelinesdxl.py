# drunkunet_sampler.py

import torch
import logging
import math
import random

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class DRUNKUNetSampler:
    """
    DRUNKUNet sampler that integrates with Fooocus ksampler.
    Applies various perturbations to the UNet's internal workings.
    """
    def __init__(self, 
                 attn_noise_strength=0.0, 
                 dynamic_guidance_params=None, # e.g., {'base': 7.0, 'amplitude': 2.0, 'frequency': 0.1, 'phase': 0}
                 layer_dropout_prob=0.0,
                 prompt_noise_strength=0.0, # Novo: Ruído no embedding textual do prompt
                 cognitive_echo_strength=0.0, # Novo: Feedback visual (eco cognitivo)
                 drunk_applied_layers=None): # e.g., ["mid", "up"] - Pode ser usado para filtrar hooks no futuro
        self.attn_noise_strength = attn_noise_strength
        self.dynamic_guidance_params = dynamic_guidance_params or {}
        self.layer_dropout_prob = layer_dropout_prob
        self.prompt_noise_strength = prompt_noise_strength
        self.cognitive_echo_strength = cognitive_echo_strength
        self.drunk_applied_layers = drunk_applied_layers or ["mid", "up"] 
        
        self.original_sampling_function = None
        self.is_active = False
        self.hook_handles = []
        self.current_step = 0 # Para tracking do step, útil para guidance dinâmico e outros
        self.global_residual_memory = None # Para o eco cognitivo

    def activate(self, unet):
        """Activate DRUNKUNet by patching the sampling function and registering hooks."""
        if self.is_active:
            return
        print(f"[DRUNKUNet] Ativando com:")
        print(f"  - Ruído de Atenção: {self.attn_noise_strength}")
        print(f"  - Guidance Dinâmico: {bool(self.dynamic_guidance_params)}")
        print(f"  - Dropout de Camada: {self.layer_dropout_prob}")
        print(f"  - Ruído no Prompt: {self.prompt_noise_strength}")
        print(f"  - Eco Cognitivo: {self.cognitive_echo_strength}")
        
        # Import the sampling module
        try:
            import ldm_patched.modules.samplers as samplers
            # Store original sampling function if not already stored
            if not hasattr(self, '_original_sampling_function'):
                self._original_sampling_function = samplers.sampling_function
            # Replace with DRUNKUNet-enhanced version
            samplers.sampling_function = self._create_drunk_sampling_function(self._original_sampling_function)
            self.unet = unet
            
            # Registrar hooks para perturbações internas (ex: ruído em atenção)
            self._register_hooks(unet)
            
            self.is_active = True
            print("[DRUNKUNet] Successfully patched sampling function and registered hooks.")
        except Exception as e:
            print(f"[DRUNKUNet] Falha ao ativar: {e}")
            # Tenta desativar parcialmente caso tenha dado erro após ativar parcialmente
            self.deactivate() 
            return

    def deactivate(self):
        """Deactivate DRUNKUNet by restoring the original sampling function and removing hooks."""
        if not self.is_active:
            return
        print("[DRUNKUNet] Desativando")

        # Restore original sampling function
        try:
            import ldm_patched.modules.samplers as samplers
            if hasattr(self, '_original_sampling_function'):
                samplers.sampling_function = self._original_sampling_function
                print("[DRUNKUNet] Successfully restored original sampling function")
        except Exception as e:
            print(f"[DRUNKUNet] Falha ao restaurar sampling function: {e}")

        # Remover hooks registrados
        try:
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()
            print("[DRUNKUNet] Hooks removidos com sucesso.")
        except Exception as e:
            print(f"[DRUNKUNet] Falha ao remover hooks: {e}")

        self.is_active = False
        self.current_step = 0
        self.global_residual_memory = None # Resetar memória do eco cognitivo

    def _create_drunk_sampling_function(self, original_sampling_function):
        """Create DRUNKUNet-modified sampling function"""
        def drunk_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
            # Armazenar o step atual para uso em hooks/guidance
            # Uma aproximação simples: assumindo que timestep diminui a cada step
            # Uma forma mais robusta seria passar o step diretamente ou usar um contador global
            # Por enquanto, vamos incrementar a cada chamada
            self.current_step += 1 
            
            # Calcular guidance scale dinâmico, se configurado
            dynamic_cond_scale = cond_scale
            if self.dynamic_guidance_params:
                 base_guidance = self.dynamic_guidance_params.get('base', cond_scale)
                 amplitude = self.dynamic_guidance_params.get('amplitude', 1.0)
                 frequency = self.dynamic_guidance_params.get('frequency', 0.1)
                 phase = self.dynamic_guidance_params.get('phase', 0)
                 # Exemplo: sin wave
                 dynamic_cond_scale = base_guidance + amplitude * math.sin(frequency * self.current_step + phase)
                 # Garantir que não seja negativo ou muito pequeno
                 dynamic_cond_scale = max(0.1, dynamic_cond_scale) 
                 # print(f"[DRUNK] Step {self.current_step}, Guidance Scale: {dynamic_cond_scale}") # Debug

            # Aplicar ruído no prompt (cond e uncond)
            if self.prompt_noise_strength > 0.0:
                if isinstance(cond, torch.Tensor):
                    cond = cond + torch.randn_like(cond) * self.prompt_noise_strength
                if isinstance(uncond, torch.Tensor):
                    uncond = uncond + torch.randn_like(uncond) * self.prompt_noise_strength
            
            # Chamar a função de amostragem original com o guidance scale (potencialmente) modificado
            # As perturbações internas (attn noise, dropout) são aplicadas pelos hooks
            try:
                return original_sampling_function(model, x, timestep, uncond, cond, dynamic_cond_scale, model_options, seed)
            except Exception as e:
                 print(f"[DRUNKUNet] Erro na sampling function, usando original: {e}")
                 return original_sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options, seed)

        return drunk_sampling_function

    def _register_hooks(self, unet):
        """Registra hooks para aplicar perturbações dentro do UNet."""
        if self.attn_noise_strength > 0.0:
             self._register_attn_noise_hooks(unet)
        if self.layer_dropout_prob > 0.0:
             self._register_dropout_hooks(unet)
        if self.cognitive_echo_strength > 0.0:
             self._register_cognitive_echo_hooks(unet)

    def _register_attn_noise_hooks(self, unet):
        """Registra hooks para adicionar ruído aos mapas de atenção."""
        def attn_noise_hook(module, input, output):
            if self.is_active and self.attn_noise_strength > 0.0:
                try:
                    # Assumindo que 'output' seja o tensor de atenção (attn_map) ou uma tupla contendo-o.
                    # A estrutura exata depende da implementação do ldm_patched.
                    # Este é um exemplo genérico. Pode precisar de ajuste.
                    if isinstance(output, torch.Tensor):
                        noise = torch.randn_like(output, device=output.device) * self.attn_noise_strength
                        return output + noise
                    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                         # Se a saída for uma tupla, assumindo que o primeiro elemento é o attn_map relevante
                         noise = torch.randn_like(output[0], device=output[0].device) * self.attn_noise_strength
                         modified_output = (output[0] + noise,) + output[1:]
                         return modified_output
                    # Adicione mais condições se a estrutura for diferente
                except Exception as e:
                    print(f"[DRUNKUNet] Erro ao aplicar ruído no hook de atenção: {e}")
            return output # Retorna a saída original se não for aplicável ou ocorrer erro

        # Iterar pelas camadas do UNet e registrar hooks nas de Cross-Attention
        # Você precisa identificar as camadas corretas. 
        # 'tpg_applied_layers' pode ser usado para filtrar 'mid', 'up', etc.
        # Este é um exemplo genérico. O nome exato pode variar ('attn2', 'CrossAttention', etc.)
        try:
            # Access the actual UNet model from the ModelPatcher
            actual_unet = unet.model if hasattr(unet, 'model') else unet
            for name, module in actual_unet.named_modules():
                 # Exemplo de filtro, ajuste conforme a estrutura real do UNet
                 if "attn2" in name: 
                     handle = module.register_forward_hook(attn_noise_hook)
                     self.hook_handles.append(handle)
            print(f"[DRUNKUNet] Registrados {len(self.hook_handles)} hooks de ruído de atenção.")
        except Exception as e:
             print(f"[DRUNKUNet] Erro ao registrar hooks de atenção: {e}")


    def _register_dropout_hooks(self, unet):
        """Placeholder para hooks de dropout em camadas."""
        # Esta é uma ideia mais complexa. Dropout normalmente é aplicado durante treinamento.
        # Para aplicar durante inferência, você poderia adicionar um hook que zera aleatoriamente
        # alguns elementos da entrada ou saída de uma camada com uma certa probabilidade.
        # Por exemplo, em uma saída de uma ResNet block:
        def layer_dropout_hook(module, input, output):
             if self.is_active and self.layer_dropout_prob > 0.0:
                 try:
                     # Exemplo: Dropout na saída
                     if isinstance(output, torch.Tensor) and random.random() < self.layer_dropout_prob:
                          # Aplica dropout manualmente
                          dropout_mask = torch.rand_like(output, device=output.device) > self.layer_dropout_prob
                          return output * dropout_mask / (1 - self.layer_dropout_prob) # Escala para compensar
                     # Pode ser aplicado a diferentes tipos de saída (tuplas etc.)
                 except Exception as e:
                     print(f"[DRUNKUNet] Erro ao aplicar dropout no hook: {e}")
             return output

        # Registrar em camadas desejadas (ResNet blocks, Transformer blocks etc.)
        # Exemplo genérico:
        try:
            for name, module in unet.named_modules():
                 # Exemplo de filtro, ajuste conforme necessário
                 if any(layer_type in name for layer_type in self.drunk_applied_layers): # Usa o filtro de camadas
                     # Pode ser necessário ser mais específico sobre quais subcamadas receberão o hook
                     if hasattr(module, 'out_channels') or 'transformer' in name.lower(): # Exemplo de condição
                          handle = module.register_forward_hook(layer_dropout_hook)
                          self.hook_handles.append(handle)
            print(f"[DRUNKUNet] Registrados hooks de dropout (se houver camadas correspondentes).")
        except Exception as e:
             print(f"[DRUNKUNet] Erro ao registrar hooks de dropout: {e}")

    def _register_cognitive_echo_hooks(self, unet):
        """Registra hooks para adicionar feedback visual (eco cognitivo) entre camadas."""
        def cognitive_echo_hook(module, input, output):
            if self.is_active and self.cognitive_echo_strength > 0.0:
                try:
                    if isinstance(output, torch.Tensor):
                        if self.global_residual_memory is not None and self.global_residual_memory.shape == output.shape:
                            output = output + self.global_residual_memory * self.cognitive_echo_strength
                        self.global_residual_memory = output.detach().clone()
                    elif isinstance(output, tuple) and len(output) > 0 and isinstance(output[0], torch.Tensor):
                        if self.global_residual_memory is not None and self.global_residual_memory.shape == output[0].shape:
                            modified_output_tensor = output[0] + self.global_residual_memory * self.cognitive_echo_strength
                            output = (modified_output_tensor,) + output[1:]
                        self.global_residual_memory = output[0].detach().clone()
                except Exception as e:
                    print(f"[DRUNKUNet] Erro ao aplicar eco cognitivo no hook: {e}")
            return output

        try:
            # Registrar em camadas estratégicas, por exemplo, após blocos principais ou antes da saída
            # Este é um exemplo genérico, ajuste conforme a arquitetura do UNet
        try:
            # Access the actual UNet model from the ModelPatcher
            actual_unet = unet.model if hasattr(unet, 'model') else unet
            for name, module in actual_unet.named_modules():
                # Exemplo: aplicar em blocos de saída ou em camadas intermediárias importantes
                if "output_blocks" in name or "mid_block" in name:
                    handle = module.register_forward_hook(cognitive_echo_hook)
                    self.hook_handles.append(handle)
            print(f"[DRUNKUNet] Registrados hooks de eco cognitivo (se houver camadas correspondentes).")
        except Exception as e:
            print(f"[DRUNKUNet] Erro ao registrar hooks de eco cognitivo: {e}")

# Instância global do DRUNKUNetSampler
drunkunet_sampler = DRUNKUNetSampler(
    attn_noise_strength=0.0,
    dynamic_guidance_params=None,
    layer_dropout_prob=0.0,
    prompt_noise_strength=0.0, # Adicionado para ruído no prompt
    cognitive_echo_strength=0.0 # Adicionado para eco cognitivo
)

# --- Placeholder para compatibilidade (similar ao TPG) ---
# Se você tiver código que importa uma classe pipeline, mantenha isso para evitar erros.

class StableDiffusionXLDRUNKUNetPipeline:
    """
    Placeholder DRUNKUNet Pipeline class for compatibility.
    The main DRUNKUNet functionality is handled by this sampler.
    """
    def __init__(self, *args, **kwargs):
        """Initialize DRUNKUNet pipeline - placeholder for compatibility"""
        pass

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """Create DRUNKUNet pipeline from pretrained model - placeholder for compatibility"""
        return cls()

    def __call__(self, *args, **kwargs):
        """DRUNKUNet pipeline call - placeholder for compatibility"""
        raise NotImplementedError("DRUNKUNet functionality is handled by the sampler integration.")
