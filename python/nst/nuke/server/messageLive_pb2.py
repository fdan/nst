# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: messageLive.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='messageLive.proto',
  package='mlserver',
  syntax='proto2',
  serialized_pb=_b('\n\x11messageLive.proto\x12\x08mlserver\"i\n\x0eRequestWrapper\x12\x0c\n\x04info\x18\x01 \x01(\x08\x12!\n\x02r1\x18\x02 \x01(\x0b\x32\x15.mlserver.RequestInfo\x12&\n\x02r2\x18\x03 \x01(\x0b\x32\x1a.mlserver.RequestInference\"\x89\x01\n\x0eRespondWrapper\x12\x0c\n\x04info\x18\x01 \x01(\x08\x12!\n\x02r1\x18\x02 \x01(\x0b\x32\x15.mlserver.RespondInfo\x12&\n\x02r2\x18\x03 \x01(\x0b\x32\x1a.mlserver.RespondInference\x12\x1e\n\x05\x65rror\x18\x04 \x01(\x0b\x32\x0f.mlserver.Error\"\x1b\n\x0bRequestInfo\x12\x0c\n\x04info\x18\x01 \x01(\x08\"B\n\x0bRespondInfo\x12\x12\n\nnum_models\x18\x01 \x01(\x05\x12\x1f\n\x06models\x18\x02 \x03(\x0b\x32\x0f.mlserver.Model\"\x8f\x03\n\x05Model\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05label\x18\x02 \x01(\t\x12(\n\x06inputs\x18\x03 \x03(\x0b\x32\x18.mlserver.ImagePrototype\x12)\n\x07outputs\x18\x04 \x03(\x0b\x32\x18.mlserver.ImagePrototype\x12*\n\x0c\x62ool_options\x18\x05 \x03(\x0b\x32\x14.mlserver.BoolAttrib\x12(\n\x0bint_options\x18\x06 \x03(\x0b\x32\x13.mlserver.IntAttrib\x12,\n\rfloat_options\x18\x07 \x03(\x0b\x32\x15.mlserver.FloatAttrib\x12.\n\x0estring_options\x18\x08 \x03(\x0b\x32\x16.mlserver.StringAttrib\x12,\n\x0e\x62utton_options\x18\t \x03(\x0b\x32\x14.mlserver.BoolAttrib\x12\x32\n\nmc_options\x18\n \x03(\x0b\x32\x1e.mlserver.MultipleChoiceOption\"D\n\x14MultipleChoiceOption\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t\x12\x0f\n\x07\x63hoices\x18\x03 \x03(\t\"0\n\x0eImagePrototype\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x10\n\x08\x63hannels\x18\x02 \x01(\x05\"\x14\n\x05\x45rror\x12\x0b\n\x03msg\x18\x01 \x01(\t\"\x93\x01\n\x10RequestInference\x12\x1e\n\x05model\x18\x01 \x01(\x0b\x32\x0f.mlserver.Model\x12\x1f\n\x06images\x18\x02 \x03(\x0b\x32\x0f.mlserver.Image\x12\x12\n\nclearcache\x18\x03 \x01(\x05\x12\x13\n\x0b\x62\x61tch_total\x18\x04 \x01(\x05\x12\x15\n\rbatch_current\x18\x05 \x01(\x05\"\x8d\x01\n\x10RespondInference\x12\x12\n\nnum_images\x18\x01 \x01(\x05\x12\x1f\n\x06images\x18\x02 \x03(\x0b\x32\x0f.mlserver.Image\x12\x13\n\x0bnum_objects\x18\x03 \x01(\x05\x12/\n\x07objects\x18\x04 \x03(\x0b\x32\x1e.mlserver.FieldValuePairAttrib\"G\n\x05Image\x12\r\n\x05width\x18\x01 \x01(\x05\x12\x0e\n\x06height\x18\x02 \x01(\x05\x12\x10\n\x08\x63hannels\x18\x03 \x01(\x05\x12\r\n\x05image\x18\x04 \x01(\x0c\".\n\nBoolAttrib\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\x06values\x18\x02 \x03(\x08\x42\x02\x10\x01\"-\n\tIntAttrib\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\x06values\x18\x02 \x03(\x05\x42\x02\x10\x01\"/\n\x0b\x46loatAttrib\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\x06values\x18\x02 \x03(\x02\x42\x02\x10\x01\",\n\x0cStringAttrib\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x0e\n\x06values\x18\x02 \x03(\t\"N\n\x14\x46ieldValuePairAttrib\x12\x0c\n\x04name\x18\x01 \x01(\t\x12(\n\x06values\x18\x02 \x03(\x0b\x32\x18.mlserver.FieldValuePair\"\xd3\x01\n\x0e\x46ieldValuePair\x12+\n\x0eint_attributes\x18\x01 \x03(\x0b\x32\x13.mlserver.IntAttrib\x12/\n\x10\x66loat_attributes\x18\x02 \x03(\x0b\x32\x15.mlserver.FloatAttrib\x12\x31\n\x11string_attributes\x18\x03 \x03(\x0b\x32\x16.mlserver.StringAttrib\x12\x30\n\x08\x63hildren\x18\x04 \x03(\x0b\x32\x1e.mlserver.FieldValuePairAttrib')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_REQUESTWRAPPER = _descriptor.Descriptor(
  name='RequestWrapper',
  full_name='mlserver.RequestWrapper',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='info', full_name='mlserver.RequestWrapper.info', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='r1', full_name='mlserver.RequestWrapper.r1', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='r2', full_name='mlserver.RequestWrapper.r2', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=31,
  serialized_end=136,
)


_RESPONDWRAPPER = _descriptor.Descriptor(
  name='RespondWrapper',
  full_name='mlserver.RespondWrapper',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='info', full_name='mlserver.RespondWrapper.info', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='r1', full_name='mlserver.RespondWrapper.r1', index=1,
      number=2, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='r2', full_name='mlserver.RespondWrapper.r2', index=2,
      number=3, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='error', full_name='mlserver.RespondWrapper.error', index=3,
      number=4, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=139,
  serialized_end=276,
)


_REQUESTINFO = _descriptor.Descriptor(
  name='RequestInfo',
  full_name='mlserver.RequestInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='info', full_name='mlserver.RequestInfo.info', index=0,
      number=1, type=8, cpp_type=7, label=1,
      has_default_value=False, default_value=False,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=278,
  serialized_end=305,
)


_RESPONDINFO = _descriptor.Descriptor(
  name='RespondInfo',
  full_name='mlserver.RespondInfo',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_models', full_name='mlserver.RespondInfo.num_models', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='models', full_name='mlserver.RespondInfo.models', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=307,
  serialized_end=373,
)


_MODEL = _descriptor.Descriptor(
  name='Model',
  full_name='mlserver.Model',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.Model.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='label', full_name='mlserver.Model.label', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='inputs', full_name='mlserver.Model.inputs', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='outputs', full_name='mlserver.Model.outputs', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='bool_options', full_name='mlserver.Model.bool_options', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='int_options', full_name='mlserver.Model.int_options', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_options', full_name='mlserver.Model.float_options', index=6,
      number=7, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='string_options', full_name='mlserver.Model.string_options', index=7,
      number=8, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='button_options', full_name='mlserver.Model.button_options', index=8,
      number=9, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='mc_options', full_name='mlserver.Model.mc_options', index=9,
      number=10, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=376,
  serialized_end=775,
)


_MULTIPLECHOICEOPTION = _descriptor.Descriptor(
  name='MultipleChoiceOption',
  full_name='mlserver.MultipleChoiceOption',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.MultipleChoiceOption.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='value', full_name='mlserver.MultipleChoiceOption.value', index=1,
      number=2, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='choices', full_name='mlserver.MultipleChoiceOption.choices', index=2,
      number=3, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=777,
  serialized_end=845,
)


_IMAGEPROTOTYPE = _descriptor.Descriptor(
  name='ImagePrototype',
  full_name='mlserver.ImagePrototype',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.ImagePrototype.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channels', full_name='mlserver.ImagePrototype.channels', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=847,
  serialized_end=895,
)


_ERROR = _descriptor.Descriptor(
  name='Error',
  full_name='mlserver.Error',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='msg', full_name='mlserver.Error.msg', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=897,
  serialized_end=917,
)


_REQUESTINFERENCE = _descriptor.Descriptor(
  name='RequestInference',
  full_name='mlserver.RequestInference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='model', full_name='mlserver.RequestInference.model', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='images', full_name='mlserver.RequestInference.images', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='clearcache', full_name='mlserver.RequestInference.clearcache', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_total', full_name='mlserver.RequestInference.batch_total', index=3,
      number=4, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='batch_current', full_name='mlserver.RequestInference.batch_current', index=4,
      number=5, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=920,
  serialized_end=1067,
)


_RESPONDINFERENCE = _descriptor.Descriptor(
  name='RespondInference',
  full_name='mlserver.RespondInference',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='num_images', full_name='mlserver.RespondInference.num_images', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='images', full_name='mlserver.RespondInference.images', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='num_objects', full_name='mlserver.RespondInference.num_objects', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='objects', full_name='mlserver.RespondInference.objects', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1070,
  serialized_end=1211,
)


_IMAGE = _descriptor.Descriptor(
  name='Image',
  full_name='mlserver.Image',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='width', full_name='mlserver.Image.width', index=0,
      number=1, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='height', full_name='mlserver.Image.height', index=1,
      number=2, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='channels', full_name='mlserver.Image.channels', index=2,
      number=3, type=5, cpp_type=1, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='image', full_name='mlserver.Image.image', index=3,
      number=4, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1213,
  serialized_end=1284,
)


_BOOLATTRIB = _descriptor.Descriptor(
  name='BoolAttrib',
  full_name='mlserver.BoolAttrib',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.BoolAttrib.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='mlserver.BoolAttrib.values', index=1,
      number=2, type=8, cpp_type=7, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1286,
  serialized_end=1332,
)


_INTATTRIB = _descriptor.Descriptor(
  name='IntAttrib',
  full_name='mlserver.IntAttrib',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.IntAttrib.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='mlserver.IntAttrib.values', index=1,
      number=2, type=5, cpp_type=1, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1334,
  serialized_end=1379,
)


_FLOATATTRIB = _descriptor.Descriptor(
  name='FloatAttrib',
  full_name='mlserver.FloatAttrib',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.FloatAttrib.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='mlserver.FloatAttrib.values', index=1,
      number=2, type=2, cpp_type=6, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=_descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1381,
  serialized_end=1428,
)


_STRINGATTRIB = _descriptor.Descriptor(
  name='StringAttrib',
  full_name='mlserver.StringAttrib',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.StringAttrib.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='mlserver.StringAttrib.values', index=1,
      number=2, type=9, cpp_type=9, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1430,
  serialized_end=1474,
)


_FIELDVALUEPAIRATTRIB = _descriptor.Descriptor(
  name='FieldValuePairAttrib',
  full_name='mlserver.FieldValuePairAttrib',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='name', full_name='mlserver.FieldValuePairAttrib.name', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='values', full_name='mlserver.FieldValuePairAttrib.values', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1476,
  serialized_end=1554,
)


_FIELDVALUEPAIR = _descriptor.Descriptor(
  name='FieldValuePair',
  full_name='mlserver.FieldValuePair',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='int_attributes', full_name='mlserver.FieldValuePair.int_attributes', index=0,
      number=1, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='float_attributes', full_name='mlserver.FieldValuePair.float_attributes', index=1,
      number=2, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='string_attributes', full_name='mlserver.FieldValuePair.string_attributes', index=2,
      number=3, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='children', full_name='mlserver.FieldValuePair.children', index=3,
      number=4, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=1557,
  serialized_end=1768,
)

_REQUESTWRAPPER.fields_by_name['r1'].message_type = _REQUESTINFO
_REQUESTWRAPPER.fields_by_name['r2'].message_type = _REQUESTINFERENCE
_RESPONDWRAPPER.fields_by_name['r1'].message_type = _RESPONDINFO
_RESPONDWRAPPER.fields_by_name['r2'].message_type = _RESPONDINFERENCE
_RESPONDWRAPPER.fields_by_name['error'].message_type = _ERROR
_RESPONDINFO.fields_by_name['models'].message_type = _MODEL
_MODEL.fields_by_name['inputs'].message_type = _IMAGEPROTOTYPE
_MODEL.fields_by_name['outputs'].message_type = _IMAGEPROTOTYPE
_MODEL.fields_by_name['bool_options'].message_type = _BOOLATTRIB
_MODEL.fields_by_name['int_options'].message_type = _INTATTRIB
_MODEL.fields_by_name['float_options'].message_type = _FLOATATTRIB
_MODEL.fields_by_name['string_options'].message_type = _STRINGATTRIB
_MODEL.fields_by_name['button_options'].message_type = _BOOLATTRIB
_MODEL.fields_by_name['mc_options'].message_type = _MULTIPLECHOICEOPTION
_REQUESTINFERENCE.fields_by_name['model'].message_type = _MODEL
_REQUESTINFERENCE.fields_by_name['images'].message_type = _IMAGE
_RESPONDINFERENCE.fields_by_name['images'].message_type = _IMAGE
_RESPONDINFERENCE.fields_by_name['objects'].message_type = _FIELDVALUEPAIRATTRIB
_FIELDVALUEPAIRATTRIB.fields_by_name['values'].message_type = _FIELDVALUEPAIR
_FIELDVALUEPAIR.fields_by_name['int_attributes'].message_type = _INTATTRIB
_FIELDVALUEPAIR.fields_by_name['float_attributes'].message_type = _FLOATATTRIB
_FIELDVALUEPAIR.fields_by_name['string_attributes'].message_type = _STRINGATTRIB
_FIELDVALUEPAIR.fields_by_name['children'].message_type = _FIELDVALUEPAIRATTRIB
DESCRIPTOR.message_types_by_name['RequestWrapper'] = _REQUESTWRAPPER
DESCRIPTOR.message_types_by_name['RespondWrapper'] = _RESPONDWRAPPER
DESCRIPTOR.message_types_by_name['RequestInfo'] = _REQUESTINFO
DESCRIPTOR.message_types_by_name['RespondInfo'] = _RESPONDINFO
DESCRIPTOR.message_types_by_name['Model'] = _MODEL
DESCRIPTOR.message_types_by_name['MultipleChoiceOption'] = _MULTIPLECHOICEOPTION
DESCRIPTOR.message_types_by_name['ImagePrototype'] = _IMAGEPROTOTYPE
DESCRIPTOR.message_types_by_name['Error'] = _ERROR
DESCRIPTOR.message_types_by_name['RequestInference'] = _REQUESTINFERENCE
DESCRIPTOR.message_types_by_name['RespondInference'] = _RESPONDINFERENCE
DESCRIPTOR.message_types_by_name['Image'] = _IMAGE
DESCRIPTOR.message_types_by_name['BoolAttrib'] = _BOOLATTRIB
DESCRIPTOR.message_types_by_name['IntAttrib'] = _INTATTRIB
DESCRIPTOR.message_types_by_name['FloatAttrib'] = _FLOATATTRIB
DESCRIPTOR.message_types_by_name['StringAttrib'] = _STRINGATTRIB
DESCRIPTOR.message_types_by_name['FieldValuePairAttrib'] = _FIELDVALUEPAIRATTRIB
DESCRIPTOR.message_types_by_name['FieldValuePair'] = _FIELDVALUEPAIR

RequestWrapper = _reflection.GeneratedProtocolMessageType('RequestWrapper', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTWRAPPER,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RequestWrapper)
  ))
_sym_db.RegisterMessage(RequestWrapper)

RespondWrapper = _reflection.GeneratedProtocolMessageType('RespondWrapper', (_message.Message,), dict(
  DESCRIPTOR = _RESPONDWRAPPER,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RespondWrapper)
  ))
_sym_db.RegisterMessage(RespondWrapper)

RequestInfo = _reflection.GeneratedProtocolMessageType('RequestInfo', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTINFO,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RequestInfo)
  ))
_sym_db.RegisterMessage(RequestInfo)

RespondInfo = _reflection.GeneratedProtocolMessageType('RespondInfo', (_message.Message,), dict(
  DESCRIPTOR = _RESPONDINFO,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RespondInfo)
  ))
_sym_db.RegisterMessage(RespondInfo)

Model = _reflection.GeneratedProtocolMessageType('Model', (_message.Message,), dict(
  DESCRIPTOR = _MODEL,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.Model)
  ))
_sym_db.RegisterMessage(Model)

MultipleChoiceOption = _reflection.GeneratedProtocolMessageType('MultipleChoiceOption', (_message.Message,), dict(
  DESCRIPTOR = _MULTIPLECHOICEOPTION,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.MultipleChoiceOption)
  ))
_sym_db.RegisterMessage(MultipleChoiceOption)

ImagePrototype = _reflection.GeneratedProtocolMessageType('ImagePrototype', (_message.Message,), dict(
  DESCRIPTOR = _IMAGEPROTOTYPE,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.ImagePrototype)
  ))
_sym_db.RegisterMessage(ImagePrototype)

Error = _reflection.GeneratedProtocolMessageType('Error', (_message.Message,), dict(
  DESCRIPTOR = _ERROR,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.Error)
  ))
_sym_db.RegisterMessage(Error)

RequestInference = _reflection.GeneratedProtocolMessageType('RequestInference', (_message.Message,), dict(
  DESCRIPTOR = _REQUESTINFERENCE,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RequestInference)
  ))
_sym_db.RegisterMessage(RequestInference)

RespondInference = _reflection.GeneratedProtocolMessageType('RespondInference', (_message.Message,), dict(
  DESCRIPTOR = _RESPONDINFERENCE,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.RespondInference)
  ))
_sym_db.RegisterMessage(RespondInference)

Image = _reflection.GeneratedProtocolMessageType('Image', (_message.Message,), dict(
  DESCRIPTOR = _IMAGE,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.Image)
  ))
_sym_db.RegisterMessage(Image)

BoolAttrib = _reflection.GeneratedProtocolMessageType('BoolAttrib', (_message.Message,), dict(
  DESCRIPTOR = _BOOLATTRIB,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.BoolAttrib)
  ))
_sym_db.RegisterMessage(BoolAttrib)

IntAttrib = _reflection.GeneratedProtocolMessageType('IntAttrib', (_message.Message,), dict(
  DESCRIPTOR = _INTATTRIB,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.IntAttrib)
  ))
_sym_db.RegisterMessage(IntAttrib)

FloatAttrib = _reflection.GeneratedProtocolMessageType('FloatAttrib', (_message.Message,), dict(
  DESCRIPTOR = _FLOATATTRIB,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.FloatAttrib)
  ))
_sym_db.RegisterMessage(FloatAttrib)

StringAttrib = _reflection.GeneratedProtocolMessageType('StringAttrib', (_message.Message,), dict(
  DESCRIPTOR = _STRINGATTRIB,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.StringAttrib)
  ))
_sym_db.RegisterMessage(StringAttrib)

FieldValuePairAttrib = _reflection.GeneratedProtocolMessageType('FieldValuePairAttrib', (_message.Message,), dict(
  DESCRIPTOR = _FIELDVALUEPAIRATTRIB,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.FieldValuePairAttrib)
  ))
_sym_db.RegisterMessage(FieldValuePairAttrib)

FieldValuePair = _reflection.GeneratedProtocolMessageType('FieldValuePair', (_message.Message,), dict(
  DESCRIPTOR = _FIELDVALUEPAIR,
  __module__ = 'messageLive_pb2'
  # @@protoc_insertion_point(class_scope:mlserver.FieldValuePair)
  ))
_sym_db.RegisterMessage(FieldValuePair)


_BOOLATTRIB.fields_by_name['values'].has_options = True
_BOOLATTRIB.fields_by_name['values']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_INTATTRIB.fields_by_name['values'].has_options = True
_INTATTRIB.fields_by_name['values']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
_FLOATATTRIB.fields_by_name['values'].has_options = True
_FLOATATTRIB.fields_by_name['values']._options = _descriptor._ParseOptions(descriptor_pb2.FieldOptions(), _b('\020\001'))
# @@protoc_insertion_point(module_scope)
