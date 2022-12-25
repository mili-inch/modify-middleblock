from modules import shared, scripts
from modules.processing import process_images
import gradio as gr


_forward = None


def _forward_curried(self, attn1_type, attn2_type):
    def _forward(x, context=None):
        x = (
            self.attn1(
                self.norm1(x), context=context if self.disable_self_attn else None
            )
            if attn1_type == 0
            else self.attn2(self.norm1(x), context=context)
        ) + x
        x = (
            self.attn2(self.norm2(x), context=context)
            if attn2_type == 1
            else self.attn1(self.norm2(x), context=None)
        ) + x
        x = self.ff(self.norm3(x)) + x
        return x

    return _forward


def hijack_unet_forward(sd_model, attn1_type, attn2_type):
    global _forward
    _forward = (
        sd_model.model.diffusion_model.middle_block[1].transformer_blocks[0]._forward
    )
    sd_model.model.diffusion_model.middle_block[1].transformer_blocks[
        0
    ]._forward = _forward_curried(
        self=sd_model.model.diffusion_model.middle_block[1].transformer_blocks[0],
        attn1_type=attn1_type,
        attn2_type=attn2_type,
    )


def undo_unet_forward(sd_model):
    global _forward
    if _forward is not None:
        sd_model.model.diffusion_model.middle_block[1].transformer_blocks[
            0
        ]._forward = _forward


class Script(scripts.Script):
    def title(self):
        return "Modify middleblock attention"

    def show(self, is_img2img):
        return True

    def ui(self, is_img2img):
        attn1_type = gr.Radio(
            choices=["self-attention", "cross-attention"],
            value="self-attention",
            type="index",
            label="attn1",
            interactive=True,
        )
        attn2_type = gr.Radio(
            choices=["self-attention", "cross-attention"],
            value="cross-attention",
            type="index",
            label="attn2",
            interactive=True,
        )
        return [attn1_type, attn2_type]

    def run(self, p, attn1_type, attn2_type):
        hijack_unet_forward(
            sd_model=shared.sd_model, attn1_type=attn1_type, attn2_type=attn2_type
        )
        proc = process_images(p)
        undo_unet_forward(shared.sd_model)

        return proc
