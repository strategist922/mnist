
require(mxnet)
require(magrittr)
require(rkdb)
require(ggplot2)

kdb <- open_connection(port=8888)
# execute(-kdb,'monitor:()')

# Network configuration

# network <- mx.symbol.Variable("data") %>% 
#  mx.symbol.FullyConnected( num_hidden=128) %>%
#  mx.symbol.Activation( act_type="relu") %>%
#  mx.symbol.FullyConnected(  num_hidden = 64) %>%
#  mx.symbol.Activation( act_type="relu") %>%
#  mx.symbol.FullyConnected( num_hidden=10) %>%
#  mx.symbol.SoftmaxOutput

network <- 
  mx.symbol.Variable('data') %>%
  mx.symbol.Convolution(kernel=c(5,5), num_filter=20) %>%
  mx.symbol.Activation(act_type="tanh") %>%
  mx.symbol.Pooling(pool_type="max", kernel=c(2,2), stride=c(2,2)) %>%
  mx.symbol.Convolution(kernel=c(5,5), num_filter=50) %>%
  mx.symbol.Activation(act_type="tanh") %>%
  mx.symbol.Pooling(pool_type="max", kernel=c(2,2), stride=c(2,2)) %>%
  mx.symbol.Flatten %>%
  mx.symbol.FullyConnected(num_hidden=100) %>%
  mx.symbol.Activation(act_type="tanh") %>%
  mx.symbol.FullyConnected(num_hidden=10) %>%
  mx.symbol.SoftmaxOutput

KdbIter <- setRefClass("KdbIter",
	fields=c("iter","hdl", "data.shape", "batch.size"),
	contains = "Rcpp_MXArrayDataIter",
	methods=list(
	 	initialize=function(hdl, data.shape, batch.size){
			.self$hdl <- hdl
			.self$iter <- 0
			.self$data.shape <- data.shape
			.self$batch.size <- batch.size
			.self
	   	},
		value=function(){
			# execute(.self$hdl,'{`env set x}',.self$env)
			val<-execute(.self$hdl,'{[iter;size] t:select from train where ind>=`long$first size*iter,ind<`long$first size*iter+1;`label`data!(t`label;raze[t`data]%255 )}',.self$iter,.self$batch.size)
			val.y <- val$label
			val.x <- val$data
			dim(val.x) <- c(28, 28, 1, batch.size)
			val.x <- mx.nd.array(val.x)
			val.y <- mx.nd.array(val.y)
			list(data=val.x, label=val.y)
		},
		iter.next=function(){
			.self$iter <- .self$iter + 1
			execute(kdb,'{[iter;size] count[train] > iter*size }',.self$iter,.self$batch.size)
		},
		reset=function(){
			.self$iter <- 0
			.self$iter
		},
		num.pad=function(){},
		finalize=function(){}
	)
)


batch.size <- 600
kdb.iter <- KdbIter$new(kdb,data.shape = 28, batch.size = batch.size) 

speedometer <- function(kdb,batch.size, frequency=50){
  id<-execute(kdb,'.upd.init[]')
  function(iteration, nbatch, env, verbose=TRUE) {

    if(!is.null(env$model) && nbatch==1){
       model <- env$model
       arg <- arguments(model$symbol)
       l=list()
       l[["id"]] <- id
       for(n in c('nbatch','iteration')){ l[[n]] <- env[[n]] }
       l[["result"]] <- env$result$value
       for(i in 2:(length(arg) - 1) ){
        v<-model$arg.params[[arg[i]]] %>% as.array
        dim(v) = NULL
        l[[arg[i]]] <- v
       }
       execute(-kdb,'.upd.monitor',l)    
  	}

    count <- nbatch
    if(is.null(env$count)) env$count <- 0
    if(is.null(env$init)) env$init <- FALSE
    if (env$count > count) env$init <- FALSE
    env$count = count
    if(env$init){      
      if (count %% frequency == 0 && !is.null(env$metric)){
        time <- as.double(difftime(Sys.time(), env$tic, units = "secs"))
        speed <- frequency*batch.size/time
        result <- env$metric$get(env$train.metric)
        if (nbatch != 0 & verbose)
          message(paste0("Batch [", nbatch, "] Speed: ", speed, " samples/sec Train-",
                     result$name, "=", result$value))
        env$tic = Sys.time()
      }      
    } else {
      env$init <- TRUE
      env$tic <- Sys.time()
    }
  }
}

tic <- proc.time()
model <- mx.model.FeedForward.create(
  symbol=network,
  X=kdb.iter,
  ctx=mx.gpu(0),
  num.round=20,
  array.batch.size=batch.size,
  learning.rate=0.1,
  momentum=0.9,  
  eval.metric=mx.metric.accuracy,
  wd=0.00001,
  batch.end.callback=speedometer(kdb, batch.size, frequency = 100)
  )
print(proc.time() - tic)


