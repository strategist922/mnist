
\d .shape

del:{![y;();1b;$[0>type x;enlist;(::)] x]}
/ melt:{raze{flip (`var_,x,`val)!enlist[z],y x,z}[x;y] each cols[y] except x}
/ melt:{r:flip raze{ flip enlist[z],y x,z}[x;z] each cols[z] except x;
/ 	   flip(y[`var],x,y[`val])!r}

melt:{(x,`variable`val) xcols ungroup flip(`variable,x,`val)!flip c,'y each x,/:c:cols[y] except x}

cast:{x:?[0>type x;enlist x;x];m:(first 1#0#)each group(y`val)!y`variable;?[y;();x!x;({z,x!y};`variable;`val;m)]}

\d .

