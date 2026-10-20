"""The descriptor pool the generated modules register Envoy's protos in, rather than protobuf's default pool.

Envoy's protos keep their own file names and packages, so in the default pool they would meet any other copy of them in the process, such as xds-protos's, and whichever was loaded second would fail. Protobuf's own types, which they import, are added from the runtime's.
"""

from google.protobuf import descriptor_pool
from google.protobuf import any_pb2
from google.protobuf import descriptor_pb2
from google.protobuf import duration_pb2
from google.protobuf import timestamp_pb2
from google.protobuf import wrappers_pb2
from google.protobuf import struct_pb2

POOL = descriptor_pool.DescriptorPool()
for _module in (any_pb2, descriptor_pb2, duration_pb2, timestamp_pb2, wrappers_pb2, struct_pb2,):
    POOL.AddSerializedFile(_module.DESCRIPTOR.serialized_pb)
