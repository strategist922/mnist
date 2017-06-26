
list.of.packages <- c("R.utils")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages, repos = "https://cloud.r-project.org/")

require(R.utils)
require(magrittr)
require(visNetwork)

#setwd(tempdir())

files=c('train-images-idx3-ubyte','train-labels-idx1-ubyte','t10k-images-idx3-ubyte','t10k-labels-idx1-ubyte')

dir.create('data')

for(file in files){
 if(file.exists(sprintf('data/%s',file))) next;
 file %>% sprintf('http://yann.lecun.com/exdb/mnist/%s.gz',.) %>% download.file(destfile=file %>% sprintf('data/%s.gz',.) )
 file %>% sprintf('data/%s.gz',.) %>% gunzip
}

require(mxnet)

# Network configuration
batch.size <- 100
network <- mx.symbol.Variable("data") %>% 
 mx.symbol.FullyConnected( num_hidden=128) %>%
 mx.symbol.Activation( act_type="relu") %>%
 mx.symbol.FullyConnected(  num_hidden = 64) %>%
 mx.symbol.Activation( act_type="relu") %>%
 mx.symbol.FullyConnected( num_hidden=10) %>%
 mx.symbol.SoftmaxOutput

dtrain = mx.io.MNISTIter(
  image="data/train-images-idx3-ubyte",
  label="data/train-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=TRUE,
  flat=TRUE,
  silent=0,
  seed=10)

dtest = mx.io.MNISTIter(
  image="data/t10k-images-idx3-ubyte",
  label="data/t10k-labels-idx1-ubyte",
  data.shape=c(784),
  batch.size=batch.size,
  shuffle=FALSE,
  flat=TRUE,
  silent=0)

mx.set.seed(0)

dir.create('saved')

# create the model
model <- mx.model.FeedForward.create(network,X=dtrain, eval.data=dtest,
                                     ctx=mx.gpu(0), num.round=1,
                                     learning.rate=0.1, momentum=0.9,
                                     initializer=mx.init.uniform(0.07),
                                     epoch.end.callback=mx.callback.save.checkpoint("saved/chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(100))


dir.create('vis')

g <- model$symbol %>% graph.viz(type='vis')  

getwd() %>% sprintf('%s/vis/network.html',.) %>% visSave(g, . , selfcontained = TRUE)



# do prediction
pred <- predict(model, dtest,ctx=mx.gpu(0))

label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))



# load the model
model <- mx.model.load("saved/chkpt", 1)

#continue training with some new arguments
tic <- proc.time()
model <- mx.model.FeedForward.create(model$symbol, X=dtrain, eval.data=dtest,
                                     ctx=mx.gpu(0), num.round=20,
                                     learning.rate=0.1, momentum=0.9,
                                     epoch.end.callback=mx.callback.save.checkpoint("saved/reload_chkpt"),
                                     batch.end.callback=mx.callback.log.train.metric(100),
                                     arg.params=model$arg.params, aux.params=model$aux.params)
print(proc.time() - tic)
# do prediction
pred <- predict(model, dtest)
label <- mx.io.extract(dtest, "label")
dataX <- mx.io.extract(dtest, "data")
# Predict with R's array
pred2 <- predict(model, X=dataX)

accuracy <- function(label, pred) {
  ypred = max.col(t(as.array(pred)))
  return(sum((as.array(label) + 1) == ypred) / length(label))
}

print(paste0("Finish prediction... accuracy=", accuracy(label, pred)))
print(paste0("Finish prediction... accuracy2=", accuracy(label, pred2)))




