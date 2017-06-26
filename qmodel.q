args:.Q.def[`name`port!("qmodel.q";8888);].Q.opt .z.x

/ remove this line when using in production
/ qmodel.q:localhost:8888::
{ if[not x=0; @[x;"\\\\";()]]; value"\\p 8888"; } @[hopen;`:localhost:8888;0];

\l shape.q

images:read1`$":data/train-images-idx3-ubyte"
label:read1`$":data/train-labels-idx1-ubyte"

dim:"j"$4 cut 12#4 _ images
dim:sum@'xexp[256;reverse til 4] */:dim
data:("j"$prd -2#dim)cut "j"$16_images
label:"h"$8 _ label
train:update ind:i from ([]label;data)

images:read1`$":data/t10k-images-idx3-ubyte"
label:read1`$":data/t10k-labels-idx1-ubyte"
dim:"j"$4 cut 12#4 _ images
dim:sum@'xexp[256;reverse til 4] */:dim
data:("j"$prd -2#dim)cut "j"$16_images
label:"h"$8 _ label
test:update ind:i from ([]label;data)

.upd.monitor:{ `monitor insert .shape.melt[`id`iteration] delete nbatch from  enlist dic: @[;`result;{[x;y]first x };()] @[;`iteration;{[x;y]first x };()] @[;`nbatch;{[x;y]first x };()] @[;`id;{[x;y]"G"$x};()] dic:(`$x . 0,`names)!x 1;
 }

.upd.init:{ string first 1?0ng}

monitor:()

/ pgs:enlist`sym`arg!(`,{})

/ .z.pg:{`pgs insert enlist`sym`arg!(`pg;x); value x}
/ .z.ps:{`pgs insert enlist`sym`arg!(`ps;x); value x}

.monitor.result:{ 
 tmp:`iteration xasc select from monitor where id ="G"$x;
 r:cols[tmp]#ungroup 0!select id,iteration,val:{[x] sqrt sum@'w*w:x - @[;0;{y#0#x};count first x]prev x } val by variable from tmp where not variable=`result;
 r,select from tmp where variable=`result}

.vis.convWeight0:{[id0;var0;dim] ungroup update xdim: {[x;y]raze ("j"$prd -2#x)#enlist til x 0}[dim]@'val,ydim:{[x;y]raze (x 2)#enlist raze (x 0 )#'til x 1}[dim]@'val,zdim:{[x;y]raze ("j"$prd 2#x)#'til x 2}[dim]@'val from select from monitor where id =id0,variable=var0 }

.vis.convWeight:{[id0;var0;dim] .vis.convWeight0["G"$id0;`$var0;"j"$dim] }

.vis.convBias:{[id0;var0] ungroup update xdim:{`$string til count x}@'val from select from monitor where id="G"$id0,variable=`$var0 }


/ 

10#monitor

/ distinct exec variable from monitor
/ `convolution2_weight`convolution2_bias`convolution3_weight`convolution3_bias`fullyconnected2_weight`fullyconnected2_bias`fullyconnected3_weight`fullyconnected3_bias

/ select {first @' x }@'val from monitor where variable =`convolution2_weight
/ select {count x}@'val from monitor where variable =`convolution2_weight

/ id0:"G"$"be698aa4-5f34-4195-bb74-7a24d2aa3cea"
/ var0:`convolution1_weight
/ dim:5 5 1000
/ .vis.convWeight["bc150163-c551-0eba-8871-9767f5c0e3d5";`convolution2_weight;5 5 20]

/ 5 cut first 25 cut v 0

/ .monitor.result "bc150163-c551-0eba-8871-9767f5c0e3d5"

/ `:qmodel.json 0: enlist symbol