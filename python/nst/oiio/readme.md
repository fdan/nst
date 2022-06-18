The oiio implementation of NST uses OpenImageIO for file IO operations, reading/writing various image formats to pytorch tensors.  

This code currently makes use of python libraries such as os, shutil, numpy, openvc, pyyaml.  A goal is to reduce these where possible to make it easier to eventually port this to c++. 