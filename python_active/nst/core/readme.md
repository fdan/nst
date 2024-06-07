This is the core nst code, which only has dependencies on pytorch / torchvision / kornia, and can subsequently be compiled to torchscript.  

The code layout is dictated by pytorch modules.  

This core code is used by each IO implementation (nuke, oiio, ofx)