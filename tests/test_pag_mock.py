import unittest
from unittest.mock import MagicMock, patch
import torch

# Mock the parent class and its dependencies
class MockStableDiffusionXLPipeline:
    def __init__(self, vae, text_encoder, text_encoder_2, tokenizer, tokenizer_2, unet, scheduler):
        self.vae = vae
        self.text_encoder = text_encoder
        self.text_encoder_2 = text_encoder_2
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.unet = unet
        self.scheduler = scheduler
        self.image_processor = MagicMock()

    def __call__(self, *args, **kwargs):
        # Simulate the parent call, returning a mock output
        mock_output = MagicMock()
        mock_output.images = [MagicMock(spec=torch.Tensor)] # Simulate a tensor image
        return mock_output

# Import the actual pipeline after mocking its parent
with patch('diffusers.StableDiffusionXLPipeline', MockStableDiffusionXLPipeline):
    from extras.PAG.pipeline_sdxl_pag import StableDiffusionXLPAGPipeline, PAGAttentionProcessor

class TestStableDiffusionXLPAGPipeline(unittest.TestCase):

    def setUp(self):
        # Mock UNet and its components
        self.mock_unet = MagicMock()
        self.mock_unet.named_children.return_value = [
            ('mid_block', MagicMock(spec=torch.nn.Module, get_processor=MagicMock(return_value=MagicMock()))),
            ('up_blocks.0', MagicMock(spec=torch.nn.Module, get_processor=MagicMock(return_value=MagicMock()))),
            ('down_blocks.0', MagicMock(spec=torch.nn.Module, get_processor=MagicMock(return_value=MagicMock()))),
        ]
        self.mock_unet.set_processor = MagicMock() # Mock the set_processor method

        # Mock other pipeline components
        self.mock_vae = MagicMock()
        self.mock_text_encoder = MagicMock()
        self.mock_text_encoder_2 = MagicMock()
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer_2 = MagicMock()
        self.mock_scheduler = MagicMock()

        self.pipeline = StableDiffusionXLPAGPipeline(
            vae=self.mock_vae,
            text_encoder=self.mock_text_encoder,
            text_encoder_2=self.mock_text_encoder_2,
            tokenizer=self.mock_tokenizer,
            tokenizer_2=self.mock_tokenizer_2,
            unet=self.mock_unet,
            scheduler=self.mock_scheduler
        )

    def test_pag_enabled_and_disabled(self):
        pag_scale = 0.5
        pag_applied_layers = ["mid", "up"]

        # Call the pipeline with PAG enabled
        print(f"[MOCK DEBUG] Calling pipeline with pag_scale={pag_scale}, pag_applied_layers={pag_applied_layers}")
        self.pipeline(
            prompt="test prompt",
            pag_scale=pag_scale,
            pag_applied_layers=pag_applied_layers
        )

        # Assert enable_pag was called
        print(f"[MOCK DEBUG] After pipeline call: do_perturbed_attention_guidance={self.pipeline.do_perturbed_attention_guidance}")
        self.assertTrue(self.pipeline.do_perturbed_attention_guidance)
        self.assertEqual(self.pipeline.pag_scale, pag_scale)
        self.assertEqual(self.pipeline._pag_applied_layers, pag_applied_layers)

        # Verify that PAGAttentionProcessor was set for relevant layers
        # This requires inspecting the calls to set_processor on the unet
        set_processor_calls = self.mock_unet.set_processor.call_args_list
        print(f"[MOCK DEBUG] Number of set_processor calls: {len(set_processor_calls)}")
        self.assertGreater(len(set_processor_calls), 0)

        # Check if PAGAttentionProcessor was used for 'mid' and 'up' blocks
        pag_processor_found = False
        for call_args, _ in set_processor_calls:
            processor = call_args[0]
            print(f"[MOCK DEBUG]   Checking processor type: {type(processor)}")
            if isinstance(processor, PAGAttentionProcessor):
                pag_processor_found = True
                print(f"[MOCK DEBUG]   Found PAGAttentionProcessor with perturbation_scale={processor.perturbation_scale}")
                self.assertEqual(processor.perturbation_scale, pag_scale) # Verify scale is passed

        self.assertTrue(pag_processor_found, "PAGAttentionProcessor was not set for any layer.")

        # Assert disable_pag was called after the generation
        print(f"[MOCK DEBUG] After disable: do_perturbed_attention_guidance={self.pipeline.do_perturbed_attention_guidance}")
        self.assertFalse(self.pipeline.do_perturbed_attention_guidance)
        self.assertEqual(self.pipeline.pag_scale, 0.0)
        self.assertEqual(self.pipeline._pag_applied_layers, [])

    def test_pag_attention_processor_perturbation(self):
        original_processor = MagicMock()
        pag_processor = PAGAttentionProcessor(original_processor, perturbation_scale=0.5)
        print(f"[MOCK DEBUG] Initialized PAGAttentionProcessor with perturbation_scale={pag_processor.perturbation_scale}")

        mock_attn = MagicMock()
        mock_attn.get_attention_scores.return_value = torch.ones(1, 1, 10, 10) # Mock attention scores

        hidden_states = torch.randn(1, 10, 10)
        encoder_hidden_states = torch.randn(1, 10, 10)

        # Test with batch size 1 (no PAG)
        print("[MOCK DEBUG] Calling PAGAttentionProcessor with batch_size=1")
        pag_processor(mock_attn, hidden_states, encoder_hidden_states)
        original_processor.assert_called_once()
        print(f"[MOCK DEBUG] original_processor called once: {original_processor.called}")
        original_processor.reset_mock()

        # Test with batch size 3 (PAG active)
        hidden_states_pag = torch.randn(3, 10, 10)
        encoder_hidden_states_pag = torch.randn(3, 10, 10)
        print("[MOCK DEBUG] Calling PAGAttentionProcessor with batch_size=3")
        pag_processor(mock_attn, hidden_states_pag, encoder_hidden_states_pag)

        # Assert original_processor was called three times (uncond, cond, perturb)
        print(f"[MOCK DEBUG] original_processor call count: {original_processor.call_count}")
        self.assertEqual(original_processor.call_count, 3)

        # Assert that get_attention_scores was temporarily replaced and called
        print(f"[MOCK DEBUG] mock_attn.get_attention_scores call count: {mock_attn.get_attention_scores.call_count}")
        self.assertGreater(mock_attn.get_attention_scores.call_count, 0)

if __name__ == '__main__':
    unittest.main()
