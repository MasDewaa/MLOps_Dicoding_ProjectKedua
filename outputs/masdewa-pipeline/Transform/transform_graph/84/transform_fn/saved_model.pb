ųG
š
Ć

8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
³
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Į
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring Ø
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
<
StringLower	
input

output"
encodingstring "serve*2.10.12v2.10.0-76-gfdfc646704c8Ó0
y
serving_default_inputsPlaceholder*'
_output_shapes
:’’’’’’’’’*
dtype0	*
shape:’’’’’’’’’
{
serving_default_inputs_1Placeholder*'
_output_shapes
:’’’’’’’’’*
dtype0*
shape:’’’’’’’’’
Ź
PartitionedCallPartitionedCallserving_default_inputsserving_default_inputs_1*
Tin
2	*
Tout
2	*
_collective_manager_ids
 *:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference_signature_wrapper_30

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ź
valueĄB½ B¶

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures* 
* 
* 
* 
* 
* 
* 

serving_default* 
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 

StatefulPartitionedCallStatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *$
fR
__inference__traced_save_55

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__traced_restore_65ę$
š
Z
 __inference_signature_wrapper_30

inputs	
inputs_1
identity	

identity_1
PartitionedCallPartitionedCallinputsinputs_1*
Tin
2	*
Tout
2	*:
_output_shapes(
&:’’’’’’’’’:’’’’’’’’’* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *
fR
__inference_pruned_20`
IdentityIdentityPartitionedCall:output:0*
T0	*'
_output_shapes
:’’’’’’’’’b

Identity_1IdentityPartitionedCall:output:1*
T0*'
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’:O K
'
_output_shapes
:’’’’’’’’’
 
_user_specified_nameinputs:QM
'
_output_shapes
:’’’’’’’’’
"
_user_specified_name
inputs_1

O
__inference_pruned_20

inputs	
inputs_1
identity	

identity_1Q
inputs_copyIdentityinputs*
T0	*'
_output_shapes
:’’’’’’’’’\
IdentityIdentityinputs_copy:output:0*
T0	*'
_output_shapes
:’’’’’’’’’U
inputs_1_copyIdentityinputs_1*
T0*'
_output_shapes
:’’’’’’’’’[
StringLowerStringLowerinputs_1_copy:output:0*'
_output_shapes
:’’’’’’’’’^

Identity_1IdentityStringLower:output:0*
T0*'
_output_shapes
:’’’’’’’’’"
identityIdentity:output:0"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:’’’’’’’’’:’’’’’’’’’:- )
'
_output_shapes
:’’’’’’’’’:-)
'
_output_shapes
:’’’’’’’’’
Å
D
__inference__traced_restore_65
file_prefix

identity_1
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHr
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B £
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 X
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: J

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix

h
__inference__traced_save_55
file_prefix
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHo
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B °
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: "µ	J
saver_filename:0StatefulPartitionedCall:0StatefulPartitionedCall_18"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*
serving_default
9
inputs/
serving_default_inputs:0	’’’’’’’’’
=
inputs_11
serving_default_inputs_1:0’’’’’’’’’5
	labels_xf(
PartitionedCall:0	’’’’’’’’’3
text_xf(
PartitionedCall:1’’’’’’’’’tensorflow/serving/predict:ę

created_variables
	resources
trackable_objects
initializers

assets
transform_fn

signatures"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
-B+
__inference_pruned_20inputsinputs_1
,
serving_default"
signature_map
ĪBĖ
 __inference_signature_wrapper_30inputsinputs_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsŖ *
 õ
__inference_pruned_20Ūt¢q
j¢g
eŖb
1
labels'$
inputs/labels’’’’’’’’’	
-
text%"
inputs/text’’’’’’’’’
Ŗ "cŖ`
0
	labels_xf# 
	labels_xf’’’’’’’’’	
,
text_xf!
text_xf’’’’’’’’’õ
 __inference_signature_wrapper_30Ši¢f
¢ 
_Ŗ\
*
inputs 
inputs’’’’’’’’’	
.
inputs_1"
inputs_1’’’’’’’’’"cŖ`
0
	labels_xf# 
	labels_xf’’’’’’’’’	
,
text_xf!
text_xf’’’’’’’’’