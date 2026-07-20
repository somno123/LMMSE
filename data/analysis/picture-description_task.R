library(ordinal)
library(emmeans)

cat("== Software ==\n")
cat(R.version.string, "\n")
for (p in c("ordinal", "emmeans", "lme4")) {
  cat(sprintf("  %s %s\n", p, as.character(packageVersion(p))))
}

d <- read.csv("picture_ratings.csv", stringsAsFactors = TRUE)
d$score <- factor(d$score, ordered = TRUE, levels = 1:5)
d$round <- factor(d$round)
d$description <- interaction(d$model, d$picture, d$round, drop = TRUE)  # 50
cat(sprintf("\nN ratings = %d; descriptions = %d; judges = %d; criteria = %d\n",
            nrow(d), nlevels(d$description), nlevels(d$judge), nlevels(d$criterion)))


report <- function(fit, fit0, label) {
  cat("\n\n########################################################\n")
  cat("## ", label, "\n")
  cat("########################################################\n")

  cat("\n-- Convergence --\n")
  cat(sprintf("  clmm convergence code = %s (0 = OK)\n", fit$optRes$convergence))
  cat(sprintf("  max |gradient|        = %.3e\n", max(abs(fit$gradient))))
  cat(sprintf("  logLik = %.3f, AIC = %.2f, npar = %d\n",
              as.numeric(logLik(fit)), AIC(fit), length(coef(fit))))

  cat("\n-- Random-effect variances --\n")
  vc <- ordinal::VarCorr(fit)
  for (nm in names(vc)) {
    v <- as.numeric(vc[[nm]])
    cat(sprintf("  %-12s variance = %.4f  (SD = %.4f)\n", nm, v, sqrt(v)))
  }

  # anova.clmm cannot resolve models fitted inside a function's scope, so the
  # likelihood-ratio test is computed directly from the two log-likelihoods.
  cat("\n-- Omnibus LRT for `model` --\n")
  ll1 <- as.numeric(logLik(fit))
  ll0 <- as.numeric(logLik(fit0))
  df_diff <- length(coef(fit)) - length(coef(fit0))
  lrt <- 2 * (ll1 - ll0)
  cat(sprintf("  logLik(reduced) = %.3f, logLik(full) = %.3f\n", ll0, ll1))
  cat(sprintf("  chi^2(%d) = %.2f, p = %.4f\n",
              df_diff, lrt, pchisq(lrt, df_diff, lower.tail = FALSE)))

  cat("\n-- Pairwise contrasts (EMMs; Holm over all 10 pairs; 95% CI) --\n")
  emm <- emmeans(fit, ~ model)
  raw <- summary(pairs(emm, adjust = "none"))
  adj <- summary(pairs(emm, adjust = "holm"))
  ci  <- confint(pairs(emm, adjust = "none"))       # unadjusted 95% CI, logit scale
  tab <- data.frame(contrast = raw$contrast,
                    Estimate = round(raw$estimate, 3),
                    SE       = round(raw$SE, 3),
                    LCL      = round(ci$asymp.LCL, 3),
                    UCL      = round(ci$asymp.UCL, 3),
                    z        = round(raw$z.ratio, 3),
                    p_raw    = round(raw$p.value, 4),
                    p_adj    = round(adj$p.value, 4))
  print(tab, row.names = FALSE)
  invisible(tab)
}


## ---- M0: original specification (criterion and judge random) ----
m0  <- clmm(score ~ model + picture + (1|description) + (1|judge) + (1|criterion),
            data = d, link = "logit")
m00 <- clmm(score ~ picture + (1|description) + (1|judge) + (1|criterion),
            data = d, link = "logit")
t0 <- report(m0, m00, "M0  ORIGINAL: (1|description) + (1|judge) + (1|criterion)")

## ---- M1: criterion FIXED (the sensitivity analysis the reviewer asked for) ----
m1  <- clmm(score ~ model + picture + criterion + (1|description) + (1|judge),
            data = d, link = "logit")
m10 <- clmm(score ~ picture + criterion + (1|description) + (1|judge),
            data = d, link = "logit")
t1 <- report(m1, m10, "M1  SENSITIVITY: criterion FIXED, (1|description) + (1|judge)")

## ---- M2: criterion and judge both FIXED (proposed primary) ----
m2  <- clmm(score ~ model + picture + criterion + judge + (1|description),
            data = d, link = "logit")
m20 <- clmm(score ~ picture + criterion + judge + (1|description),
            data = d, link = "logit")
t2 <- report(m2, m20, "M2  PROPOSED PRIMARY: criterion + judge FIXED, (1|description)")

