

graph.viz(model$symbol,type='vis')

id <-  execute(kdb,'last distinct exec id from monitor')

monitor <- execute(kdb,'.monitor.result', id)

ggplot(monitor,aes(iteration,val,color=variable)) + geom_point() + geom_line() 


var <- execute(kdb,'{distinct exec variable except `result  from monitor where id="G"$x}',id)

tmp<- execute(kdb,'.vis.convWeight',id,var[1],c(5,5,20))

ggplot(tmp,aes(xdim,ydim,fill=val))+ geom_tile() + scale_fill_gradient2(low = "blue",  high = "red") + facet_grid(iteration ~ zdim )

tmp <- execute(kdb,'.vis.convBias',id,var[4])

ggplot(tmp,aes(iteration,val,color=xdim)) + geom_point()+ geom_line()


tmp<- execute(kdb,'{select from .vis.convWeight[x;y;z] where zdim within "j"$0 20 + 20*12}',id,var[3],c(5,5,20*50))

ggplot(tmp,aes(xdim,ydim,fill=val))+ geom_tile() + scale_fill_gradient2(low = "blue", high = "red") + facet_grid(iteration ~ zdim )







