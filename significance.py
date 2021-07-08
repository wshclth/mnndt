#!/usr/bin/env python
"""
Hypothesis testing is crucial to understing if you have something worth wild
ir if you should find a job and buy SPY.
"""
from matplotlib import pyplot as plt
from numpy import linalg
from scipy.stats import chi2
from scipy.spatial import distance

import yfinance
import numpy as np
import pandas

class SignificanceTester:
    """
        SignificanceTester is a hypothesis testing technique to verify that
        the returns of a strategy are statistically signifiant. In other words,
        verify that the returns of your strategy are indeed outperforming
        the market in a significant way.

        Most signifiance testing happens in 1D. This signifiance testing
        works by computing the ND "normal" distribution of the log adjusted
        returns for N assets. A byproduct of this is that corrilations between
        assets become accounted for. For example, trading two corrilated assets
        and producing similar returns on both of them will not increase your
        signifiance because the returns are already corrilated. For 1D 2-sigma
        is used as the cutoff. For anything past 1D, the chi-squared inverse
        cdf value equivilent to 95% is used as a cutoff for the
        mahalanobis distance between the strategies μ-vector and the assets
        μ-vector. If your strategy produces greater than 2-sigma returns in 1D,
        or the mahalanobis distance is greater than the chi-squared cutoff for ND
        then your strategy does indeed produce signifiant returns compared the
        underlying assets that it trades.
    """
    def __init__(self, start_time):
        """
            Initializes a new test given a known start time of your strategy

            params:
                start_time - The start time your strategy started.
        """
        self.syms = []
        self.cov_data = []
        self.us = []
        self.start_time = start_time

    def add_returns(self, asset):
        """
            Adds the returns of an asset to the test.

            params:
                asset - The asset to include in the signifiance test.
        """
        asset_data = yfinance.download(asset, self.start_time, progress=False)
        return_data = np.log1p(asset_data['Adj Close'].pct_change())
        self.us.append(np.mean(return_data))
        self.cov_data.append(return_data[1:])
        self.syms.append(asset)


    # Nondegenerate Multivariate Normal Distribution.
    # Helper function that computes the nondegenerate multivariate normal
    # distribution.
    def _ndmnd(self, x, us, precision, generalized_variance):
        x = np.array(x)
        numerator = np.exp(-0.5 * (x - us) * precision * np.transpose(x - us))
        denominator = np.sqrt(((2*np.pi)**len(x)) * generalized_variance)
        return numerator / denominator

    # Mahalanobis distance
    # Helper function to compute the mahalanobis distance of a given u vector
    def _mahalanobis_distance(self, x, us, precision):
        x = np.array(x)
        return distance.mahalanobis(x, us, precision)

    def _die_value(self):
        if len(self.syms) == 1:
            return 2
        else:
            return chi2.ppf(0.95, len(self.syms)).item()

    def test(self, mu, visualize=True):
        """
            Performs the signifiance test

            params:
                mu - Your average daily returns
                visualize - Visualizes the data if possible
        """
        us = np.matrix(self.us)
        sigmas = np.cov(self.cov_data)

        # Invert sigma to obtain the precision matrix and obtain the generalized
        # variance
        if sigmas.shape == ():
            precision = 1.0/sigmas
            generalized_variance = sigmas
        else:
            lambda_, v = np.linalg.eig(sigmas)
            if not np.all(lambda_ > 0):
                print('distibution is degenerate, test failed.')
                print('degenerate distribution happen when the cov matrix ' + \
                      'is not positive definite.')
                exit(1)

            # This will never have an error since the above confirmed that all
            # eigvalues are positive
            lambda_ = np.sqrt(lambda_)

            # Attempt to obtain the precision matrix by inverting the cov
            # matrix.
            try:
                precision = linalg.inv(sigmas)
            except np.linalg.LinAlgError:
                print('matrix is singular. did you add two of the same asset?')
                exit(1)
            generalized_variance = linalg.det(sigmas)

        die_value = self._die_value()

        if visualize:
            if len(self.syms) == 1:
                fig, ax = plt.subplots(figsize=(17.7, 10))
                ax.set_title('1D Significance Cutoff')
                xs = np.linspace(min(self.cov_data[-1]), max(self.cov_data[-1]), 500)
                ys = [self._ndmnd([x], us, precision, generalized_variance).item() for x in xs]
                ax.hist(self.cov_data[-1], density=True, alpha=0.5)
                ax.plot(xs, ys, label='density')
                ax.set_xlabel(self.syms[-1] + ' log returns')
                ax.set_ylabel('density')
                ax.axvline(us.item() + (die_value*np.sqrt(sigmas.item())), label=r'$\mu$-cutoff', color='black')
                ax.axvline(mu[0], label='$\mu$-strategy', color='red')
                plt.legend()
                plt.show()
            elif len(self.syms) == 2:
                fig, ax = plt.subplots(figsize=(17.7, 10))
                ax.set_title('2D Significance Cutoff')
                ax.set_xlabel('%s log-returns (cov adj density)' % self.syms[0])
                ax.set_ylabel('%s log-returns (cov adj density)' % self.syms[1])
                colors = []
                tx = []
                ty = []
                for x in self.cov_data[0]:
                    for y in self.cov_data[1]:
                        # Translate data into cartisian plane
                        sc = v * np.matrix([[x],[y]])
                        tx.append(sc[0].item())
                        ty.append(sc[1].item())
                        if self._mahalanobis_distance([x,y], us, precision) < die_value:
                            colors.append('black')
                        else:
                            colors.append('red')
                ax.scatter(tx, ty, alpha=0.5, color='white', edgecolors=colors)

                space = np.linspace(0, 2*np.pi, 1000)
                xs = []
                ys = []
                for s in space:
                    # Compute ellipse based on cutoff
                    elv = v * np.matrix([[die_value*(lambda_[0])*np.cos(s)],[die_value*(lambda_[1])*np.sin(s)]])

                    # Apply v again as the transformation matrix to cartisian plane
                    elv = v * elv
                    xs.append(elv[0].item())
                    ys.append(elv[1].item())

                tus = v * np.transpose(us)
                tmus = v * mu
                ax.axvline(tmus[0,0].item(), c='orange', lw=1, label='$\mu$(%s) (cov adj)' % self.syms[0])
                ax.axhline(tmus[1,0].item(), c='orange', lw=1, label='$\mu$(%s) (cov adj)' % self.syms[1])
                ellipse = ax.plot(xs, ys, lw=1, c='black', label=r'$\chi^2(5.991) \approx 0.95$ = Mahalanobis Distance')
                ax.legend()
                plt.show()

        return die_value, self._mahalanobis_distance(mu, us, precision)
