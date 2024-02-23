import torch
from . import initialization as init


class SegmentationModel(torch.nn.Module):
    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def check_input_shape(self, x):

        h, w, d = x.shape[-3:]
        if self.encoder.strides is not None:
            hs, ws, ds = 1, 1, 1
            for stride in self.encoder.strides:
                hs *= stride[0]
                ws *= stride[1]
                ds *= stride[2]
            if h % hs != 0 or w % ws != 0 or d % ds != 0:
                new_h = (h // hs + 1) * hs if h % hs != 0 else h
                new_w = (w // ws + 1) * ws if w % ws != 0 else w
                new_d = (d // ds + 1) * ds if d % ds != 0 else d
                raise RuntimeError(
                    f"Wrong input shape height={h}, width={w}, depth={d}. Expected image height and width and depth "
                    f"divisible by {hs}, {ws}, {ds}. Consider pad your images to shape ({new_h}, {new_w}, {new_d})."
                )
        else:
            output_stride = self.encoder.output_stride
            if h % output_stride != 0 or w % output_stride != 0 or d % output_stride != 0:
                new_h = (h // output_stride + 1) * output_stride if h % output_stride != 0 else h
                new_w = (w // output_stride + 1) * output_stride if w % output_stride != 0 else w
                new_d = (d // output_stride + 1) * output_stride if d % output_stride != 0 else d
                raise RuntimeError(
                    f"Wrong input shape height={h}, width={w}, depth={d}. Expected image height and width and depth "
                    f"divisible by {output_stride}. Consider pad your images to shape ({new_h}, {new_w}, {new_d})."
                )

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""

        self.check_input_shape(x)

        features = self.encoder(x)
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    @torch.no_grad()
    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        x = self.forward(x)

        return x
