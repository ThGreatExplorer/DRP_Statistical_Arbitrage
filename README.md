# Statistical Arbitrage in Federal Bond Markets
By Daniel Yu and Nezam Jazayeri 

## Quickstart
[[Statistical_Arbitrage_in_Federal_Bond_Markets_v1.pdf]]

## Abstract
By nature, arbitrage opportunities are transient and concentrated in underlooked sectors. This research was motivated by the prevalence and history of trading models built on statistical arbitrage techniques in more traditional, liquid financial markets namely public equities, commodities, foreign exchange, and other derivative markets compared to the perceived lack of such exploration in the bond market. Our paper examines Federal Bond markets and indices for statistical arbitrage opportunities across different timescales. We outline the theoretical framework of Bond markets and their differentiation from equity and other financial markets given their unique characteristics such as correlation to federal interest rate, maturity date, yield, etc. Finally, we apply pairs trading techniques and construct basic trading strategies to try and generate alpha from statistical arbitrage methods. We find that ultimately, while the federal bond market is generally efficient, opportunities for statistical arbitrage exist.

## Data Collection
[Bloomberg Methodology](Bonds_Exploration.pdf)

[BBG Excels](Data/BBG_Excels/) <br>
[CSV](Data/csv/)

## Strategies
[Pair Trading](nbs/data.py) <br>
[Volatility Trading](nbs/research.long_most_volatile_post_high_volatility.nb2.ipynb)

## References
1. https://www.statsmodels.org/devel/_modules/statsmodels/tsa/stattools.html
2. https://github.com/QuantConnect/Tutorials
3. https://gist.github.com/ThGreatExplorer/ce3f69f810d86a8cf9dc12e87d79a967
4. https://link.springer.com/book/10.1007/978-1-4419-6876-0 (Specifically Chapters 2,4,5, and 7. Read Chapter 3 to review Probability and Statistics)