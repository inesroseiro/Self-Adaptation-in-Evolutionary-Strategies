#bloxplot creation - schwefel
rm(list=ls())
setwd("/Users/iroseiro/Desktop/CE_TP6/ANALYSIS/")
df = read.table("standart/schwefel.txt")
data_used_s = unlist(df,use.names = FALSE)
data_used_s
a <-table(data_used_s)
a
s_mean = mean(data_used_s)
s_std = sd(data_used_s, na.rm=FALSE)

df = read.table("standart/griewangk.txt")
data_used_g = unlist(df,use.names = FALSE)
data_used_g
a <-table(data_used_g)
a
g_mean = mean(data_used_g)
g_std = sd(data_used_g, na.rm=FALSE)

df = read.table("standart/new_quartic.txt")
data_used_q = unlist(df,use.names = FALSE)
data_used_q
a <-table(data_used_q)
a
q_mean = mean(data_used_q)
q_std = sd(data_used_q, na.rm=FALSE)

df = read.table("standart/step.txt")
data_used_st = unlist(df,use.names = FALSE)
data_used_st
a <-table(data_used_st)
a
st_mean = mean(data_used_st)
st_std = sd(data_used_st, na.rm=FALSE)

# -------------------- adaptive ---------------------------

df = read.table("adaptive/schwefel.txt")
data_used_s_a = unlist(df,use.names = FALSE)
data_used_s_a
a <-table(data_used_s_a)

s_mean_a = mean(data_used_s_a)
s_std_a = sd(data_used_s_a, na.rm=FALSE)

df = read.table("adaptive/griewangk.txt")
data_used_g_a = unlist(df,use.names = FALSE)
data_used_g_a
a <-table(data_used_g_a)

g_mean_a = mean(data_used_g_a)
g_std_a = sd(data_used_g_a, na.rm=FALSE)

df = read.table("adaptive/new_quartic.txt")
data_used_q_a = unlist(df,use.names = FALSE)
data_used_q_a
a <-table(data_used_q_a)

q_mean_a = mean(data_used_q_a)
q_std_a = sd(data_used_q_a, na.rm=FALSE)

df = read.table("adaptive/step.txt")
data_used_st_a = unlist(df,use.names = FALSE)
data_used_st_a
a <-table(data_used_st_a)

st_mean = mean(data_used_st_a)
st_std = sd(data_used_st_a, na.rm=FALSE)

# -------------- intervalos de erro -----------------------
n=500
error_s <- qt(0.975,df=n-1)*s_std/sqrt(n)
error_g <- qt(0.975,df=n-1)*g_std/sqrt(n)
error_r <- qt(0.975,df=n-1)*q_std/sqrt(n)
error_st <- qt(0.975,df=n-1)*st_std/sqrt(n)


# -----------------------------------------------------------
# From the output, the p-value > 0.05 implying that the distribution of the data are not significantly different from normal distribution. In other words, we can assume the normality.
#Feito para todos os datasets
shapiro.test(data_used_s)

# -----------------------------------------------------------
df = read.table("aux1.txt")
names(df)[names(df) == "V1"] <- "generation_best"

matriz_states <- rep(c("Standart", "Adaptive"), each = 501)

my_data <- data.frame(
   df,
   matriz_states
)

# independent 2-group Wilcoxon Signed Rank Test
wilcox.test(my_data$generation_best ~ my_data$matriz_states, data = my_data, alternative = "two.sided")


# Kruskal Wallis Test One Way Anova by Ranks
#kruskal.test(my_data$generation_best ~ my_data$matriz_states) # where y1 is numeric and A is a factor


library("ggpubr")
ggboxplot(my_data, x = "matriz_states", y = "generation_best", 
color = "matriz_states", palette = c("#00AFBB", "#E7B800"),
ylab = "Weight", xlab = "Groups", main="Best for Generation - Step")

grid()
dev.off()

# ----------------------------------------------------

#auxiliary plots

plot(data_used_s, type="l", col="blue", ylim=c(100,1000),lwd=2)


lines(data_used_s_a, type="l", lty=1, col="red",lwd=2)

title(main="Best for Generation - Schwefel", col.main="Black", font.main=2)
legend(1, 200, legend=c("Adaptive", "Standart"),
       col=c("red", "blue"), lty=1:1, cex=0.8)



plot(data_used_g, type="l", col="blue", ylim=c(0,30),lwd=2)


lines(data_used_g_a, type="l", lty=1, col="red",lwd=2)

title(main="Best for Generation - Griewangk", col.main="Black", font.main=2)
legend(400, 30, legend=c("Adaptive", "Standart"),
       col=c("red", "blue"), lty=1:1, cex=0.8)


plot(data_used_q, type="l", col="blue", ylim=c(0,1),lwd=2)


lines(data_used_q_a, type="l", lty=1, col="red",lwd=2)

title(main="Best for Generation - Quartic", col.main="Black", font.main=2)
legend(400, 1, legend=c("Adaptive", "Standart"),
       col=c("red", "blue"), lty=1:1, cex=0.8)


plot(data_used_st, type="l", col="blue", ylim=c(0,15),lwd=2)


lines(data_used_st_a, type="l", lty=1, col="red",lwd=2)

title(main="Best for Generation - Step", col.main="Black", font.main=2)
legend(400, 15, legend=c("Adaptive", "Standart"),
       col=c("red", "blue"), lty=1:1, cex=0.8)


# ---------------------------------------------------------------------
df = read.table("standart/new_quartic.txt")
data_used_q = unlist(df,use.names = FALSE)

df = read.table("standart/quartic.txt")
data_used_q_new = unlist(df,use.names = FALSE)

plot(data_used_q, type="l", col="blue", ylim=c(0,5),lwd=2)


lines(data_used_q_new, type="l", lty=1, col="red",lwd=2)

title(main="Standart - Quartic ", col.main="Black", font.main=2)
legend(400, 5, legend=c("20 individuals", "5 individuals"),
       col=c("red", "blue"), lty=1:1, cex=0.8)