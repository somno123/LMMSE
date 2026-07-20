library(ordinal)
library(emmeans)

d <- read.csv("picture_ratings.csv", stringsAsFactors = TRUE)
d$score <- factor(d$score, ordered = TRUE, levels = 1:5)
d$round <- factor(d$round)
d$description <- interaction(d$model, d$picture, d$round, drop = TRUE)  # 50 responses

##  CLMM: model fixed; description/judge/criterion random intercepts ----
fit  <- clmm(score ~ model + picture + (1|description) + (1|judge) + (1|criterion),
             data = d, link = "logit")
fit0 <- clmm(score ~ picture + (1|description) + (1|judge) + (1|criterion),
             data = d, link = "logit")

cat("== Omnibus likelihood-ratio test for `model` ==\n")
print(anova(fit0, fit))          # chi^2(4), p  -> paste into manuscript [TBD]

cat("\n== Pairwise contrasts (EMMs, Holm over all 10 pairs) ==\n")
emm <- emmeans(fit, ~ model)

ctr_raw <- summary(pairs(emm, adjust = "none"))
ctr_adj <- summary(pairs(emm, adjust = "holm"))
tab <- data.frame(contrast = ctr_raw$contrast,
                  Estimate = round(ctr_raw$estimate, 4),
                  SE       = round(ctr_raw$SE, 3),
                  z        = round(ctr_raw$z.ratio, 3),
                  p_raw    = round(ctr_raw$p.value, 4),
                  p_adj    = round(ctr_adj$p.value, 4))
print(tab, row.names = FALSE)        # 10 rows -> Table tab:desc_pairwise

