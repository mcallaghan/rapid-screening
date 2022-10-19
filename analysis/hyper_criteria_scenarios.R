library(foreach)
library(doParallel)

N_tot <- 60000 # total documents
r_tot <- N_tot*0.01 # total relevant documents


docs <- rep(0,N_tot)
docs[1:r_tot] <- 1
docs <- sample(docs, replace=F)

par(pty="s")
plot(
  cumsum(docs), 
  xlab="Documents Screened", 
  ylab="Relevant documents identified",
  xlim=c(0,N_tot),
  ylim=c()
)


tau_target=0.95
tau <- cumsum(docs)/r_tot
which(tau>tau_target)[1]/N_tot

dev.off()
par(mfrow = c(1,3),pty="s")



get_ktar <- function(r_al, r_seen, recall_target){
  return(floor(r_seen/recall_target-r_al+1))
}

h0_p <- function(docs, N_tot, recall_target, verbose=FALSE) {
  r_seen <- sum(docs)
  n_vec <- seq(1:length(docs))
  k_vec <- cumsum(rev(docs))
  r_al_vec <- r_seen - k_vec
  k_hat_vec <- vapply(r_al_vec, get_ktar, numeric(1), 
                      recall_target=recall_target, r_seen=r_seen)
  red_ball_vec <- N_tot-(length(docs)-n_vec)-k_hat_vec
  p_vec <- phyper(k_vec, k_hat_vec, red_ball_vec, n_vec)
  #p_vec <- phyperPeizer(k_vec, k_hat_vec, red_ball_vec, n_vec)
  n <- which.min(p_vec)
  if (verbose) {
    return(list("min_p" = p_vec[n], "n" = n, "p_vec" = p_vec))
  } else {
    return(p_vec[n])
  }
}

ptm <- proc.time()

df <- data.frame(
  i=seq(1,N_tot), r_seen=cumsum(ordered_docs), 
  docs=ordered_docs, pmin=0
)

doc_sets <- list()

cl <- makeCluster(4) #not to overload your computer
registerDoParallel(cl)

for (w in c(15,10,5)) {
  weights = rep(1,N_tot)
  weights[which(docs==1)] <- rep(
    w,length.out=r_tot
  )
  ordered_docs <- sample(
    docs, prob=weights, replace=F
  )
  plot(cumsum(ordered_docs))
  abline(h=r_tot*.95)
  
  tau <- cumsum(ordered_docs)/r_tot
  print(which(tau>=tau_target)[1]/N_tot)
  name <- paste0("Weight: ",w)
  doc_sets[[name]] <- list(weight=w, docs=ordered_docs)
  
  df[paste0("docs_",w)] <- ordered_docs
  
  for (t in c(.75,.9,.95)) {
    df[paste0("w_",w,"p_",t)] <- foreach(i=df$i, .combine="c") %dopar% {
      h0_p(ordered_docs[1:i], N_tot, t, verbose=F)
    }
  }
}
stopCluster(cl)



#dev.new(width=7, height=7, unit="in",noRStudioGD = TRUE)
pdf(file="~/scenarios.pdf",width=7.2, height=7)
mar <- 3
par(mfrow = c(3,3),oma = c(5,4,0,0) + 0.1,
    mar = c(0,0,2,1) + 0.1,
    pty="s")
for (w in c(15,10,5)) {
  ordered_docs <- df[[paste0("docs_",w)]]
  if (w==5) {xaxt <- NULL} else {xaxt <- "n"}
  for (t in c(.75,.9,.95)) {
    if (t==.75) {
      yaxt <- NULL
      ylab <- "Recall/p H0"
    } else {
      yaxt <- "n"
      ylab <- ""
    }
    plot(cumsum(ordered_docs)/sum(ordered_docs),
      type="l",main=paste0(t*100,"% recall target"),yaxt=yaxt,xaxt=xaxt,
      ylab=ylab)
    lines(df[[paste0("w_",w,"p_",t)]])
    tau <- cumsum(ordered_docs)/r_tot
    pf <- 1-which(tau>t)[1]/N_tot
    text(47500,0.8,paste0("Max WS\n",round(pf*100),"%"))
    p50 <- 1-which(df[[paste0("w_",w,"p_",t)]]<0.5)[1]/N_tot
    text(47500,0.5,paste0("H0<50%:\n",round(p50*100),"% WS"))
    p5 <- 1-which(df[[paste0("w_",w,"p_",t)]]<0.05)[1]/N_tot
    text(47500,0.2,paste0("H0<5%\n",round(p5*100),"% WS"))
  }
}
title(xlab = "Documents screened",
      ylab = "Recall/p H0",
      outer = TRUE, line = 3)
dev.off()


lines(df$pmin75)
tau <- cumsum(ordered_docs)/r_tot
pf <- 1-which(tau>0.75)[1]/N_tot
text(40000,0.8,paste0("Target reached with \n",round(pf*100),"% work savings (WS)"))
p50 <- 1-which(df$pmin75<0.5)[1]/N_tot
text(40000,0.5,paste0("Target likely (>50%)\n reached: ",round(p50*100),"% (WS)"))
p5 <- 1-which(df$pmin75<0.05)[1]/N_tot
text(40000,0.2,paste0("Target very likely (>95%)\nreached: ",round(p5*100),"% (WS)"))




################################



df$pmin <- foreach(i=df$i, .combine="c") %dopar% {
  h0_p(df$docs[1:i], N_tot, .95, verbose=F)
}


proc.time() - ptm

par(mfrow = c(2,1), mar = c(2, 2, 2, 2),pty="s")
plot(df$r_seen)
plot(df$pmin)



ptm <- proc.time()

cl <- makeCluster(4) #not to overload your computer
registerDoParallel(cl)

df$pmin90 <- foreach(i=df$i, .combine="c") %dopar% {
  h0_p(df$docs[1:i], N_tot, .90, verbose=F)
}
stopCluster(cl)

proc.time() - ptm

par(mfrow = c(2,1), mar = c(2, 2, 2, 2),pty="s")
plot(df$r_seen)
plot(df$pmin90)

##################


ptm <- proc.time()

cl <- makeCluster(4) #not to overload your computer
registerDoParallel(cl)

df$pmin75 <- foreach(i=df$i, .combine="c") %dopar% {
  h0_p(df$docs[1:i], N_tot, .75, verbose=F)
}
stopCluster(cl)

proc.time() - ptm

par(mfrow = c(2,1), mar = c(2, 2, 2, 2),pty="s")
plot(df$r_seen)
plot(df$pmin75)


par(mfrow = c(1,3))
plot(df$r_seen/r_tot,type="l",main="75% recall target")
lines(df$pmin75)
tau <- cumsum(ordered_docs)/r_tot
pf <- 1-which(tau>0.75)[1]/N_tot
text(40000,0.8,paste0("Target reached with \n",round(pf*100),"% work savings (WS)"))
p50 <- 1-which(df$pmin75<0.5)[1]/N_tot
text(40000,0.5,paste0("Target likely (>50%)\n reached: ",round(p50*100),"% (WS)"))
p5 <- 1-which(df$pmin75<0.05)[1]/N_tot
text(40000,0.2,paste0("Target very likely (>95%)\nreached: ",round(p5*100),"% (WS)"))


plot(df$r_seen/r_tot,type="l",main="90% recall target")
lines(df$pmin90)
pf <- 1-which(tau>0.9)[1]/N_tot
text(40000,0.8,paste0("Target reached with \n",round(pf*100),"% work savings"))
p50 <- 1-which(df$pmin90<0.5)[1]/N_tot
text(40000,0.5,paste0("Target likely (>50%)\n reached: ",round(p50*100),"% (WS)"))
p5 <- 1-which(df$pmin90<0.05)[1]/N_tot
text(40000,0.2,paste0("Target very likely (>95%)\nreached: ",round(p5*100),"% (WS)"))

plot(df$r_seen/r_tot,type="l",main="95% recall target")
lines(df$pmin)
pf <- 1-which(tau>0.95)[1]/N_tot
text(40000,0.8,paste0("Target reached with \n",round(pf*100),"% work savings"))
p50 <- 1-which(df$pmin<0.5)[1]/N_tot
text(40000,0.5,paste0("Target likely (>50%)\n reached: ",round(p50*100),"% (WS)"))
p5 <- 1-which(df$pmin<0.05)[1]/N_tot
text(40000,0.2,paste0("Target very likely (>95%)\nreached: ",round(p5*100),"% (WS)"))

