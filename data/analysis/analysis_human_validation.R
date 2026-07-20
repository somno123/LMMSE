set.seed(0)
N_BOOT <- 10000

MODELS <- c("Sonar", "Claude Sonnet 4.5", "Grok 4.1", "Gemini 3 Flash", "GPT-5.2")
PICS <- c("cookie_theft", "cat_rescue")

# rows = raters R1-R5, cols = models (paper order); cell = mean over 5 criteria
COOKIE <- matrix(c(
  4.4, 4.4, 3.4, 3.8, 4.4,
  4.0, 4.2, 3.6, 4.4, 4.4,
  3.2, 4.0, 2.6, 2.4, 4.0,
  4.2, 4.6, 3.4, 3.4, 4.2,
  4.2, 2.8, 3.8, 4.2, 2.8
), nrow = 5, ncol = 5, byrow = TRUE, dimnames = list(NULL, MODELS))

CAT <- matrix(c(
  4.6, 2.8, 3.6, 4.0, 4.4,
  4.8, 4.0, 4.0, 3.4, 4.0,
  4.0, 3.8, 3.2, 3.2, 4.0,
  3.4, 3.6, 4.0, 3.8, 3.0,
  4.4, 4.4, 4.2, 2.2, 3.8
), nrow = 5, ncol = 5, byrow = TRUE, dimnames = list(NULL, MODELS))

# human matrix: 10 units (5 models x 2 pictures) x 5 raters
# first 5 rows = cookie units, then cat
HUM <- rbind(t(COOKIE), t(CAT))


# ICC(2,1) and ICC(2,k): two-way random effects, absolute agreement.
# mat = units (rows) x raters (cols); returns a named length-2 vector.
icc2 <- function(mat) {
  n <- nrow(mat)
  k <- ncol(mat)
  grand <- mean(mat)
  row_m <- rowMeans(mat)
  col_m <- colMeans(mat)
  ss_r <- k * sum((row_m - grand)^2)
  ss_c <- n * sum((col_m - grand)^2)
  resid <- mat - outer(row_m, rep(1, k)) - outer(rep(1, n), col_m) + grand
  ss_e <- sum(resid^2)
  msr <- ss_r / (n - 1)
  msc <- ss_c / (k - 1)
  mse <- ss_e / ((n - 1) * (k - 1))
  c(icc21 = (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
    icc2k = (msr - mse) / (msr + (msc - mse) / n))
}


# Krippendorff's alpha, interval metric; mat = units x raters.
kripp_interval <- function(mat) {
  vals <- as.vector(mat)
  m <- ncol(mat)
  # observed disagreement: mean squared difference within units
  do_num <- sum(apply(mat, 1, function(row) sum(dist(row)^2)))
  do_den <- nrow(mat) * m * (m - 1) / 2
  do <- do_num / do_den
  # expected disagreement: mean squared difference over all value pairs
  diffs <- outer(vals, vals, "-")^2
  de <- sum(diffs) / (length(vals) * (length(vals) - 1))
  1 - do / de
}


# Nonparametric bootstrap over units (rows). `stat` names one element of fn's
# result when fn returns a vector; NULL when it returns a scalar.
boot_ci <- function(mat, fn, stat = NULL) {
  idx <- seq_len(nrow(mat))
  out <- numeric(0)
  for (b in seq_len(N_BOOT)) {
    s <- sample(idx, size = length(idx), replace = TRUE)
    v <- tryCatch(fn(mat[s, , drop = FALSE]), error = function(e) NULL)
    if (is.null(v) || !all(is.finite(v))) next
    out <- c(out, if (is.null(stat)) v else v[[stat]])
  }
  unname(quantile(out, c(0.025, 0.975)))
}


fmt <- function(x, d = 2) formatC(x, format = "f", digits = d)


cat("== Per-model human means (5 raters x 2 pictures) ==\n")
for (i in seq_along(MODELS)) {
  v <- mean(c(COOKIE[, i], CAT[, i]))
  cat(sprintf("  %-18s %s\n", MODELS[i], fmt(v)))
}
cat(sprintf("  overall human mean = %s; per-picture: cookie %s, cat %s\n",
            fmt(mean(HUM)), fmt(mean(COOKIE)), fmt(mean(CAT))))
cat("  per-rater means:",
    fmt((rowMeans(COOKIE) + rowMeans(CAT)) / 2), "\n")

cat("\n== Human-human reliability (10 outputs x 5 raters) ==\n")
ic <- icc2(HUM)
ci21 <- boot_ci(HUM, icc2, "icc21")
ci2k <- boot_ci(HUM, icc2, "icc2k")
cat(sprintf("  ICC(2,1) = %s  95%% bootstrap CI [%s, %s]\n",
            fmt(ic[["icc21"]]), fmt(ci21[1]), fmt(ci21[2])))
cat(sprintf("  ICC(2,k) = %s  95%% bootstrap CI [%s, %s]\n",
            fmt(ic[["icc2k"]]), fmt(ci2k[1]), fmt(ci2k[2])))
a <- kripp_interval(HUM)
cia <- boot_ci(HUM, kripp_interval)
cat(sprintf("  Krippendorff alpha (interval) = %s  95%% CI [%s, %s]\n",
            fmt(a), fmt(cia[1]), fmt(cia[2])))
dif <- unlist(lapply(seq_len(nrow(HUM)), function(u) as.vector(dist(HUM[u, ]))))
cat(sprintf("  within 1 point (rater pairs, n=%d) = %s%%; mean |diff| = %s\n",
            length(dif), fmt(mean(dif <= 1) * 100, 0), fmt(mean(dif))))

cat("\n== Human-judge agreement (10 model x picture units) ==\n")
df <- read.csv("picture_ratings.csv", stringsAsFactors = FALSE)
# judge panel mean per model x picture (over judges, rounds, criteria)
unit_judge <- aggregate(score ~ model + picture, data = df, FUN = mean)
# per-judge mean per model x picture (over rounds and criteria)
per_judge <- aggregate(score ~ model + picture + judge, data = df, FUN = mean)
JUDGES <- sort(unique(df$judge))

# units in paper order: model-major, picture-minor
units <- expand.grid(picture = PICS, model = MODELS, stringsAsFactors = FALSE)
units <- units[, c("model", "picture")]

hum_col <- function(model, picture) {
  i <- match(model, MODELS)
  if (picture == "cookie_theft") COOKIE[, i] else CAT[, i]
}
lookup <- function(tab, model, picture) {
  tab$score[tab$model == model & tab$picture == picture]
}

hp <- mapply(function(m, p) mean(hum_col(m, p)), units$model, units$picture)
jp <- mapply(function(m, p) lookup(unit_judge, m, p), units$model, units$picture)
cat("  unit, human panel, judge panel:\n")
for (u in seq_len(nrow(units))) {
  cat(sprintf("    %-18s %-12s %s  %s\n",
              units$model[u], units$picture[u], fmt(hp[u]), fmt(jp[u])))
}

off <- jp - hp
cat(sprintf(paste0("  judge - human offset = %+.2f points ",
                   "(human panel range %s-%s, judge panel range %s-%s)\n"),
            mean(off), fmt(min(hp)), fmt(max(hp)), fmt(min(jp)), fmt(max(jp))))
cat(sprintf("  Pearson r (panel means, 10 units) = %s\n", fmt(cor(hp, jp))))
two <- cbind(hp, jp)
a2 <- kripp_interval(two)
ci2 <- boot_ci(two, kripp_interval)
cat(sprintf("  Krippendorff alpha (interval, panel means) = %s  95%% CI [%s, %s]\n",
            fmt(a2), fmt(ci2[1]), fmt(ci2[2])))

# 10 x 3 judges, and 10 x 5 humans; all 150 human-judge rater pairs
jm <- t(mapply(function(m, p) {
  sub <- per_judge[per_judge$model == m & per_judge$picture == p, ]
  sub$score[match(JUDGES, sub$judge)]
}, units$model, units$picture))
hm <- t(mapply(function(m, p) hum_col(m, p), units$model, units$picture))
pair_diff <- numeric(0)
for (u in seq_len(nrow(units))) {
  pair_diff <- c(pair_diff, as.vector(outer(jm[u, ], hm[u, ], "-")))
}
cat(sprintf(paste0("  human-judge rater pairs (n=%d): within 1 pt = %s%%; ",
                   "mean judge-human = %+.2f; ",
                   "within 1 pt after removing mean offset = %s%%\n"),
            length(pair_diff),
            fmt(mean(abs(pair_diff) <= 1) * 100, 0),
            mean(pair_diff),
            fmt(mean(abs(pair_diff - mean(pair_diff)) <= 1) * 100, 0)))
