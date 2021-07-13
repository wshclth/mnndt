## Multivariate Non-degenerate Normal Distrubtion Hypothesis Tester
MNNDHT provides an answer to a common question asked in quantitative finance. Are my returns significant?
Common tests for asserting statistical signifiance include the t-test, z-score, and p-value. MNNDHT is
similar to these tests but focuses on asset distributions more closely. MNNDHT looks at the distributions
of returns of N stocks, and compares the distribution of your strategies returns on those N stocks together.


This package is focused towards finance but the general idea can be applied to anything that follows
normal distributions. Because log returns of asset prices are normally distributed, (proof can be shown by solving brownian motion), this method can be applied to financial data.


Why does this look at distribution? If the distribution of your strategies returns, mimiks the distribution
of the asset it trades on, then buying and holding provides the same returns as your strategy, or in other
words, insignificant. Being different than the returns of an asset, or the returns of multiple assets combines, means that you are no longer following the distribution of the asset, for better or for worse...

Examples:

[Example](./examples.ipynb)

References:

[Multivate Normal Distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)

[Mahalanobis Distance](https://en.wikipedia.org/wiki/Mahalanobis_distance)

[Chi-square Distribution](https://en.wikipedia.org/wiki/Chi-square_distribution)
