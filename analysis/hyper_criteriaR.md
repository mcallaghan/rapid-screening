Active learning stopping criteria
================

In an active learning pipeline for screening citations, humans screen documents, machines learn from these screening decisions what a relevant document looks like, and humans keep screening the documents predicted by the machine to be most relevant.

If machine-learning can identify relevant documents effectively, human screeners would screen all or most of the relevant documents in a set of documents before screening all or most of that set of documents. In principle they could stop at that point and save themselves some work.

Because we don't know *a priori* the number of relevant documents, human screeners won't know when they have reached a safe place to stop.

One way to approach this problem is to develop a null hypothesis that a given recall target has been missed, and to see how likely this is. If we get a p-value for this test that is below a given threshold, human screeners could stop at a level of risk of missing documents they are comfortable with.

We can calculate a p-value for this null hypothesis in R with the following information:

``` r
# A vector of already screened documents
screened <- c(rep(1,20), rep(0,70))

# The total number of documents
N <- 100

# The recall target
recall_target <- 0.9
```

To test this hypothesis we use the hypergeometric distribution, which is based on an urn of randomly mixed red and green balls, which we draw out one by one, without replacing them. In this analogy, green marbles are relevant documents, red marbles are irrelevant documents.

What we need to do first is to imagine the smallest number of green marbles that could be in the urn that would mean that our recall target had not yet been reached.

``` r
# How many relevant documents have been seen
r_seen <- sum(screened)

# let's treat the last n documents as a random sample
n <- 70
sample <- tail(screened, n)

# How many relevant documents have been in seen a random sample (or set of documents we are happy to assume does not have a systematically lower proportion of relevant documents)
x <- sum(sample)

# How many relevant documents were seen before this sample or psuedosample began
r_ML <- r_seen - x

# what's the minimum number of relevant documents there would have to be in the unseen documents + the sample in order for our target to be missed
get_khat <- function(r_ML, r_seen, recall_target){
  return(floor(r_seen/recall_target-r_ML+1))
}

k_hat <- get_khat(r_ML, r_seen, recall_target)

print(k_hat)
```

    ## [1] 3

Now we can see what the chances are of pulling out the number of green marbles from the urn that we did in our sample, given an urn with the amount of green marbles that would mean that our target was missed

``` r
p <- phyper(x, k_hat, N-(length(screened)-length(sample))-k_hat, length(sample))
print(p)
```

    ## [1] 0.001460565

In a real application, we want to test this for all past urn drawings (the last 1 draws, the last 2 draws, the last 3 draws, etc.) and find the minimum probability of our null hypothesis, and the number of balls drawn. If we can assume that the proportion of green marbles in that draw is greater than or equal to the proportion of green marbles left in the urn, then this p-value is conservative.

``` r
h0_p <- function(screened, N, recall_target) {
  
  # How many relevant documents have been seen
  r_seen <- sum(screened)
  
  # what's the minimum number of relevant documents there would have to be in the 
  # unseen documents + the sample in order for our target to be missed
  get_khat <- function(r_ML, r_seen, recall_target){
    return(floor(r_seen/recall_target-r_ML+1))
  }
  
  # Sample size
  k_vec <- seq(1:length(screened))
  # number of relevant documents in sample
  x_vec <- cumsum(rev(screened))
  # number of relevant documents seen before sampling
  r_ML_vec <- r_seen - x_vec
  
  # k_hat over sample sizes
  k_hat_vec <- vapply(r_ML_vec, get_khat, numeric(1), recall_target=recall_target, r_seen=r_seen)
  
  # corresponding irrelevant documents in sample+unseen documents
  n_vec <- N-(length(screened)-k_vec)-k_hat_vec
  
  p_vec <- phyper(x_vec, k_hat_vec, n_vec, k_vec) 
  
  n <- which.min(p_vec)
  
  return(list("min_p" = p_vec[n], "n" = n, "p_vec" = p_vec))
}

h0 <- h0_p(screened, N, recall_target)

print(paste0("Based on the last ",h0$n, " documents, the p-value for our null hypothesis that our recall is below ", recall_target, " is ", h0$min_p))
```

    ## [1] "Based on the last 70 documents, the p-value for our null hypothesis that our recall is below 0.9 is 0.00146056475170399"
