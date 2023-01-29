n <- 5
eps = sqrt(.Machine$double.eps)
X <- matrix(seq(0, 2*pi, length=n), ncol=1)
y <- sin(X)
D <- plgp::distance(X) 
Sigma <- exp(-D) + diag(eps, ncol(D))
XX <- matrix(seq(-.1, 2*pi + .1, length=100), ncol=1)
DXX <- plgp::distance(XX)
SXX <- exp(-DXX) + diag(eps, ncol(DXX))
DX <- plgp::distance(XX, X)
SX <- exp(-DX) 
Si <- solve(Sigma)
mup <- SX %*% Si %*% y
Sigmap <- SXX - SX %*% Si %*% t(SX)
YY <- SimDesign::rmvnorm(100, mup, Sigmap)
q1 <- mup + qnorm(0.05, 0, sqrt(diag(Sigmap)))
q2 <- mup + qnorm(0.95, 0, sqrt(diag(Sigmap)))
matplot(XX, t(YY), type="l", col="gray", lty=1, xlab="x", ylab="y")
points(X, y, pch=20, cex=2)
lines(XX, sin(XX), col="blue")
lines(XX, mup, lwd=2)
lines(XX, q1, lwd=2, lty=2, col=2)
lines(XX, q2, lwd=2, lty=2, col=2)

nx <- 20
x <- seq(0, 10, length=nx)
X <- expand.grid(x, x)
D <- plgp::distance(X)
Sigma <- exp(-D) + diag(eps, nrow(X))
Y <- SimDesign::rmvnorm(4, sigma=Sigma)
par(mfrow=c(1,1)) 
persp(x, x, matrix(Y[1,], ncol=nx), theta=-30, phi=30, xlab="x1", 
      ylab="x2", zlab="y",col="blue")
persp(x, x, matrix(Y[2,], ncol=nx), theta=-30, phi=30, xlab="x1", 
      ylab="x2", zlab="y",col="green")
persp(x, x, matrix(Y[3,], ncol=nx), theta=-30, phi=30, xlab="x1", 
      ylab="x2", zlab="y",col="red")
persp(x, x, matrix(Y[4,], ncol=nx), theta=-30, phi=30, xlab="x1", 
      ylab="x2", zlab="y",col="purple")


library(lhs) 
X <- randomLHS(40, 2)
X[,1] <-1:40
X[,2] <-1:40
y <- X[,1]*exp(-X[,1]^2 - X[,2]^2)
xx <- seq(0, 40, length=40)
XX <- expand.grid(xx, xx)
D <- plgp::distance(X)
Sigma <- exp(-D)
DXX <- plgp::distance(XX)
SXX <- exp(-DXX) + diag(eps, ncol(DXX))
DX <- plgp::distance(XX, X)
SX <- exp(-DX)
Si <- solve(Sigma)
mup <- SX %*% Si %*% y
Sigmap <- SXX - SX %*% Si %*% t(SX)
sdp <- sqrt(diag(Sigmap))
par(mfrow=c(1,2))
cols <- heat.colors(128)
image(xx, xx, matrix(mup, ncol=length(xx)), xlab="x1", ylab="x2", col=cols)
points(X[,1], X[,2])
image(xx, xx, matrix(sdp, ncol=length(xx)), xlab="x1", ylab="x2", col=cols)
points(X[,1], X[,2])


#observations
library(GauProMod)
library(plot3D)
library(RColorBrewer)
gamma=.85
obs <- list(x = cbind(c(1, 3, 5, 7, 9),
                      c(8, 2, 6, 4, 8)),
            y = c(5,5*gamma,5*gamma^2,5*gamma^3,5*gamma^4,
                  0,-5,-5*gamma,-5*gamma^2,-5*gamma^3,
                  0,0,2,2*gamma,2*gamma^2,
                  0,0,0,3,3*gamma,
                  0,0,0,0,10),
            t = seq_len(5))

# targets
vx <- seq(0, 10, by = 0.5)
vy <- seq(0, 10, by = 0.5)
targ <- list(x = vecGrid(vx, vy))
covModels <- list(pos =  list(kernel="gaussian",
                              l = 4,       # correlation length
                              v = 2.5,     # smoothness
                              h = 2.45),    # std. deviation
                  time = list(kernel="gaussian",
                              l = 0.15,   # correlation length
                              h = 1.25))

# 2D mean linear mean function 
op <- 2

# Gaussian likelihood
sigma <- 0.2
GP <- gpCond(obs = obs, targ = targ, covModels = covModels, 
             sigma = sigma, op = op)
names(GP)
# GP$mean   = mean value at location xstar
# GP$cov    = covariance matrix of the conditioned GP
# GP$logLik = log-likelihood of the conditioned GP
# GP$xstar  = x-coordinates at which the GP is simulated
Ymean <-  array(GP$mean, dim=c(length(obs$t), length(vx), length(vy)))
Ysd <-  array(sqrt(diag(GP$cov)), dim=c(length(obs$t), length(vx), length(vy)))

par(mfrow = c(2,5))
for(i in seq_along(obs$t)){
  mysubtitle = paste("reward =",obs$y[i*6-5])
  plot3D::image2D(z = Ymean[i,,], x = vx, y = vy, zlim = range(Ymean), 
                  main = paste("mean at t =",obs$t[i]))
  mtext(mysubtitle,side=3,line=0,cex=0.9)
  points(obs$x[i,1],obs$x[i,2], col="white",pch=20, cex=2)
  points(obs$x[i,1],obs$x[i,2], col="black",pch=3)
}

#par(mfrow = c(2,5))
for(i in seq_along(obs$t)){
  plot3D::image2D(z = Ysd[i,,], x = vx, y = vy, zlim = range(Ysd), 
                  main =  paste("std. dev. at t =",obs$t[i]))
  points(obs$x[i,1],obs$x[i,2], col="white",pch=20, cex=2)
  points(obs$x[i,1],obs$x[i,2], col="black",pch=3)
}


####################
library(ggplot2)
library(dplyr)
library(tidyr)
library(faux)

# targets
vx <- seq(0, 10, by = 1)
vy <- seq(0, 10, by = 1)
targ <- list(x = vecGrid(vx, vy))
covModels <- list(pos =  list(kernel="gaussian",
                              l = 4,       # correlation length
                              v = 2.5,     # smoothness
                              h = 2.45),    # std. deviation
                  time = list(kernel="gaussian",
                              l = 0.15,   # correlation length
                              h = 1.25))
# 2D mean linear mean function 
op <- 2
rs = c(1,2,3,4,5,6,7,8,9,10,
       2,1,1,2,3,4,5,6,7,8,
       3,2,1,2,3,4,6,7,8,9,
       4,3,2,1,1,2,3,4,5,6,
       5,3,1,2,3,4,5,6,7,8,
       6,5,5,4,3,2,3,4,5,9,
       7,7,6,6,5,4,3,3,5,6,
       8,7,6,5,4,3,2,1,4,7,
       9,8,7,7,5,3,1,4,5,8,
       10,9,8,7,6,5,4,3,2,1)
gamma=.85
n_obs = 20
reward = matrix(rs,10,10)
obs_init = rep(0,100)
x1obs = sample(1:100,n_obs,replace = F)

for(i in 1:n_obs){
  obs_init = obs_init*gamma
  obs_init[x1obs[i]] = rs[x1obs[i]]
  t = seq_len(i)
}
targ <- list(x = vecGrid(vx, vy))
obs = list(x=cbind(targ$x),
           y = obs_init,
           t=t)
# Gaussian likelihood
sigma <- 0.2
GP <- gpCond(obs = obs, targ = targ, covModels = covModels, 
             sigma = sigma, op = op)
Ymean <-  array(GP$mean, dim=c(length(obs$t), length(vx), length(vy)))
Ysd <-  array(sqrt(diag(GP$cov)), dim=c(length(obs$t), length(vx), length(vy)))

par(mfrow = c(2,5))
for(i in seq_along(obs$t)){
  mysubtitle = paste("reward =",obs$y[x1obs[i],x2obs[i]])
  plot3D::image2D(z = Ymean[i,,], x = vx, y = vy, zlim = range(Ymean), 
                  main = paste("mean at t =",obs$t[i]))
  mtext(mysubtitle,side=3,line=0,cex=0.9)
  points(obs$x[i,1],obs$x[i,2], col="white",pch=20, cex=2)
  points(obs$x[i,1],obs$x[i,2], col="black",pch=3)
}

#par(mfrow = c(2,5))
for(i in seq_along(obs$t)){
  plot3D::image2D(z = Ysd[i,,], x = vx, y = vy, zlim = range(Ysd), 
                  main =  paste("std. dev. at t =",obs$t[i]))
  points(obs$x[i,1],obs$x[i,2], col="white",pch=20, cex=2)
  points(obs$x[i,1],obs$x[i,2], col="black",pch=3)
}
