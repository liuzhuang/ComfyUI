ComfyUI
=======
最强大且模块化的 Stable Diffusion 的 GUI 和后端。
-----------
![ComfyUI Screenshot](comfyui_screenshot.png)

该用户界面将允许您使用基于图形/节点/流程图的界面设计和执行高级 stable diffusion 流水线。要查看一些工作流示例并了解 ComfyUI 的功能，请访问：
### [ComfyUI 示例](https://comfyanonymous.github.io/ComfyUI_examples/)

### [安装 ComfyUI](#installing)

## 特性
- 节点/图形/流程图界面来试验和创建不需要编程的复杂 Stable Diffusion 工作流程。
- 完全支持 SD1.x, SD2.x, [SDXL](https://comfyanonymous.github.io/ComfyUI_examples/sdxl/) and [Stable Video Diffusion](https://comfyanonymous.github.io/ComfyUI_examples/video/)
- 异步队列系统
- 许多优化：在工作流执行之间只重新执行已更改的部分。
- 命令行选项： ```--lowvram``` 使其可以在少于 3GB vram 的GPU上工作(在vram较低的GPU上自动启用) 
- 即使没有 GPU 也可以工作，使用 ```--cpu``` (slow)
- 可以加载 ckpt、safetensors 和 diffusers models/checkpoints。独立的 VAE 和 CLIP 模型。
- Embeddings/Textual inversion
- [Loras (regular, locon and loha)](https://comfyanonymous.github.io/ComfyUI_examples/lora/)
- [Hypernetworks](https://comfyanonymous.github.io/ComfyUI_examples/hypernetworks/)
- 从生成的 PNG 文件中加载完整的工作流 (带种子)。
- 将工作流程为 Json 文件保存/加载。
- 节点界面可用于创建复杂的工作流程，如 [Hires fix](https://comfyanonymous.github.io/ComfyUI_examples/2_pass_txt2img/) 或更高级的工作流程。
- [Area Composition](https://comfyanonymous.github.io/ComfyUI_examples/area_composition/)
- [Inpainting](https://comfyanonymous.github.io/ComfyUI_examples/inpaint/) 包括常规和修补模型。
- [ControlNet and T2I-Adapter](https://comfyanonymous.github.io/ComfyUI_examples/controlnet/)
- [Upscale Models (ESRGAN, ESRGAN variants, SwinIR, Swin2SR, etc...)](https://comfyanonymous.github.io/ComfyUI_examples/upscale_models/)
- [unCLIP Models](https://comfyanonymous.github.io/ComfyUI_examples/unclip/)
- [GLIGEN](https://comfyanonymous.github.io/ComfyUI_examples/gligen/)
- [Model Merging 模型合并](https://comfyanonymous.github.io/ComfyUI_examples/model_merging/)
- [LCM models and Loras](https://comfyanonymous.github.io/ComfyUI_examples/lcm/)
- [SDXL Turbo](https://comfyanonymous.github.io/ComfyUI_examples/sdturbo/)
- Latent previews with [TAESD](#how-to-show-high-quality-previews)
- 启动非常快速。
- 完全脱机工作：永远不会下载任何内容。
- [配置文件 extra_model_paths](extra_model_paths.yaml.example) 来设置模型的搜索路径。

可以在[示例页面](https://comfyanonymous.github.io/ComfyUI_examples/)上找到工作流示例

## 快捷键

| Keybind                   | Explanation                                                                                                        |
|---------------------------|--------------------------------------------------------------------------------------------------------------------|
| Ctrl + Enter              | 对当前图形进行排队生成                                                                              |
| Ctrl + Shift + Enter      | 将当前图形作为首要进行生成                                                                     |
| Ctrl + Z/Ctrl + Y         | 撤消/重做                                                                                                          |
| Ctrl + S                  | 保存工作流程                                                                                                      |
| Ctrl + O                  | 加载工作流程                                                                                                     |
| Ctrl + A                  | 选择所有节点                                                                                                   |
| Alt + C                   | 折叠/展开所选节点                                                                                 |
| Ctrl + M                  | Mute/unmute 所选节点                                                                                     |
| Ctrl + B                  | 绕过所选节点(就像节点从图形中删除并重新连接线一样)            |
| Delete/Backspace          | 删除所选节点                                                                                              |
| Ctrl + Delete/Backspace   | 删除当前图                                                                                           |
| 空格                       | 按住时移动光标移动画布                                                             |
| Ctrl/Shift + Click        | 将单击的节点添加到选择集                                                                                      |
| Ctrl + C/Ctrl + V         | 复制和粘贴选定的节点(不保持与未选定节点输出的连接)                    |
| Ctrl + C/Ctrl + Shift + V | 复制和粘贴选定的节点(维护从未选定节点的输出到粘贴节点的输入的连接) |
| Shift + Drag              | 同时移动多个选定的节点                                                                      |
| Ctrl + D                  | 加载默认图                                                                                                 |
| Q                         | 切换队列的可见性                                                                                     |
| H                         | 切换历史记录的可见性                                                                                      |
| R                         | 刷新图                                                                                                      |
| 双击鼠标左键          | 打开节点快速搜索面板                                                                                     |

对于 macOS 用户 Ctrl 也可以替换为 Cmd 

# 安装

## Windows

There is a portable standalone build for Windows that should work for running on Nvidia GPUs or for running on your CPU only on the [releases page](https://github.com/comfyanonymous/ComfyUI/releases).

### [Direct link to download](https://github.com/comfyanonymous/ComfyUI/releases/download/latest/ComfyUI_windows_portable_nvidia_cu121_or_cpu.7z)

Simply download, extract with [7-Zip](https://7-zip.org) and run. Make sure you put your Stable Diffusion checkpoints/models (the huge ckpt/safetensors files) in: ComfyUI\models\checkpoints

If you have trouble extracting it, right click the file -> properties -> unblock

#### How do I share models between another UI and ComfyUI?

See the [Config file](extra_model_paths.yaml.example) to set the search paths for models. In the standalone windows build you can find this file in the ComfyUI directory. Rename this file to extra_model_paths.yaml and edit it with your favorite text editor.

## Jupyter Notebook

To run it on services like paperspace, kaggle or colab you can use my [Jupyter Notebook](notebooks/comfyui_colab.ipynb)

## Manual Install (Windows, Linux)

Git clone this repo.

Put your SD checkpoints (the huge ckpt/safetensors files) in: models/checkpoints

Put your VAE in: models/vae


### AMD GPUs (Linux only)
AMD users can install rocm and pytorch with pip if you don't have it already installed, this is the command to install the stable version:

```pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7```

This is the command to install the nightly with ROCm 6.0 which might have some performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.0```

### NVIDIA

Nvidia users should install stable pytorch using this command:

```pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121```

This is the command to install pytorch nightly instead which might have performance improvements:

```pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu121```

#### Troubleshooting

If you get the "Torch not compiled with CUDA enabled" error, uninstall torch with:

```pip uninstall torch```

And install it again with the command above.

### Dependencies

Install the dependencies by opening your terminal inside the ComfyUI folder and:

```pip install -r requirements.txt```

After this you should have everything installed and can proceed to running ComfyUI.

### Others:

#### [Intel Arc](https://github.com/comfyanonymous/ComfyUI/discussions/476)

#### Apple Mac silicon

You can install ComfyUI in Apple Mac silicon (M1 or M2) with any recent macOS version.

1. Install pytorch nightly. For instructions, read the [Accelerated PyTorch training on Mac](https://developer.apple.com/metal/pytorch/) Apple Developer guide (make sure to install the latest pytorch nightly).
1. Follow the [ComfyUI manual installation](#manual-install-windows-linux) instructions for Windows and Linux.
1. Install the ComfyUI [dependencies](#dependencies). If you have another Stable Diffusion UI [you might be able to reuse the dependencies](#i-already-have-another-ui-for-stable-diffusion-installed-do-i-really-have-to-install-all-of-these-dependencies).
1. Launch ComfyUI by running `python main.py --force-fp16`. Note that --force-fp16 will only work if you installed the latest pytorch nightly.

> **Note**: Remember to add your models, VAE, LoRAs etc. to the corresponding Comfy folders, as discussed in [ComfyUI manual installation](#manual-install-windows-linux).

#### DirectML (AMD Cards on Windows)

```pip install torch-directml``` Then you can launch ComfyUI with: ```python main.py --directml```

### I already have another UI for Stable Diffusion installed do I really have to install all of these dependencies?

You don't. If you have another UI installed and working with its own python venv you can use that venv to run ComfyUI. You can open up your favorite terminal and activate it:

```source path_to_other_sd_gui/venv/bin/activate```

or on Windows:

With Powershell: ```"path_to_other_sd_gui\venv\Scripts\Activate.ps1"```

With cmd.exe: ```"path_to_other_sd_gui\venv\Scripts\activate.bat"```

And then you can use that terminal to run ComfyUI without installing any dependencies. Note that the venv folder might be called something else depending on the SD UI.

# 运行

```python main.py```

### For AMD cards not officially supported by ROCm

Try running it with this command if you have issues:

For 6700, 6600 and maybe other RDNA2 or older: ```HSA_OVERRIDE_GFX_VERSION=10.3.0 python main.py```

For AMD 7600 and maybe other RDNA3 cards: ```HSA_OVERRIDE_GFX_VERSION=11.0.0 python main.py```

# 注意

Only parts of the graph that have an output with all the correct inputs will be executed.

Only parts of the graph that change from each execution to the next will be executed, if you submit the same graph twice only the first will be executed. If you change the last part of the graph only the part you changed and the part that depends on it will be executed.

Dragging a generated png on the webpage or loading one will give you the full workflow including seeds that were used to create it.

You can use () to change emphasis of a word or phrase like: (good code:1.2) or (bad code:0.8). The default emphasis for () is 1.1. To use () characters in your actual prompt escape them like \\( or \\).

You can use {day|night}, for wildcard/dynamic prompts. With this syntax "{wild|card|test}" will be randomly replaced by either "wild", "card" or "test" by the frontend every time you queue the prompt. To use {} characters in your actual prompt escape them like: \\{ or \\}.

Dynamic prompts also support C-style comments, like `// comment` or `/* comment */`.

To use a textual inversion concepts/embeddings in a text prompt put them in the models/embeddings directory and use them in the CLIPTextEncode node like this (you can omit the .pt extension):

```embedding:embedding_filename.pt```


## How to increase generation speed?

Make sure you use the regular loaders/Load Checkpoint node to load checkpoints. It will auto pick the right settings depending on your GPU.

You can set this command line setting to disable the upcasting to fp32 in some cross attention operations which will increase your speed. Note that this will very likely give you black images on SD2.x models. If you use xformers or pytorch attention this option does not do anything.

```--dont-upcast-attention```

## How to show high-quality previews?

Use ```--preview-method auto``` to enable previews.

The default installation includes a fast latent preview method that's low-resolution. To enable higher-quality previews with [TAESD](https://github.com/madebyollin/taesd), download the [taesd_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesd_decoder.pth) (for SD1.x and SD2.x) and [taesdxl_decoder.pth](https://github.com/madebyollin/taesd/raw/main/taesdxl_decoder.pth) (for SDXL) models and place them in the `models/vae_approx` folder. Once they're installed, restart ComfyUI to enable high-quality previews.

## Support and dev channel

[Matrix space: #comfyui_space:matrix.org](https://app.element.io/#/room/%23comfyui_space%3Amatrix.org) (it's like discord but open source).

# QA

### Why did you make this?

I wanted to learn how Stable Diffusion worked in detail. I also wanted something clean and powerful that would let me experiment with SD without restrictions.

### Who is this for?

This is for anyone that wants to make complex workflows with SD or that wants to learn more how SD works. The interface follows closely how SD works and the code should be much more simple to understand than other SD UIs.
