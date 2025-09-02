## Video Mask2Former Demo

We provide a command line tool to run a simple demo of builtin configs.
The usage is explained in [GETTING_STARTED.md](../GETTING_STARTED.md).

The demo supports input frames of arbitrary size. Frames are automatically
padded to the DINOv2 patch size (14×14) during inference and the padding is
stripped from the predictions before visualization.