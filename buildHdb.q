args:.Q.def[`name`port!("buildHdb.q";8888);].Q.opt .z.x

/ remove this line when using in production
/ buildHdb.q:localhost:8888::	
{ if[not x=0; @[x;"\\\\";()]]; value"\\p 8888"; } @[hopen;`:localhost:8888;0];

{system "l ",getenv[ `btLibrary],"\\",string[x],".q"}@'`bt`rinit

r) require(ggplot2)

`:data/sample_submission.csv

test:(784#"i";", ") 0:`:data/test.csv
train:(785#"i";", ") 0:`:data/train.csv
train:`ind xcols update ind:i from train
train0:{(`ind`label#x),enlist[`pixel]!enlist value `ind`label _ x  }@'train
test:`ind xcols update ind:i from test
test0:{(enlist[`ind]#x),enlist[`pixel]!enlist value enlist[`ind] _ x  }@'test

.upd.monitor:{`monitor insert update `$id,first@'nbatch,first@'iteration,first@'result  from enlist (`$x . 0,`names)!x 1}


/ pgs: enlist `sym`arg!(`;{})

/ .z.pg:{`pgs insert enlist `sym`arg!(`pg;x);value x }
/ .z.ps:{`pgs insert enlist `sym`arg!(`ps;x);value x }

/
cols monitor
`id`nbatch`iteration`result`convolution0_weight`convolution0_bias`convolution1_weight`convolution1_bias`fullyconnected0_weight`fullyconnected0_bias`fullyconnected1_weight`fullyconnected1_bias

reverse select id,nbatch,iteration,result,convolution0_weight:{[x;y] sqrt sum@'w*w:x - y}[ {@[x;0;{[x;y] y};count[ last x]#0nf ]} prev convolution0_weight;convolution0_weight] from monitor

tmp:select id,nbatch,iteration,result,convolution0_weight:{[x;y] sqrt sum@'w*w:x - y}[ {@[x;0;{[x;y] y};count[ last x]#0nf ]} prev convolution0_weight;convolution0_weight] from monitor

p) ggplot(`tmp,aes(iteration,convolution0_weight)) + geom_point()