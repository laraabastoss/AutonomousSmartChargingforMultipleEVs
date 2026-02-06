# Smart Charging for Multiple EVs using a Supervised Learning Based Approach

With the growing adoption of electric vehicles, the development of efficient smart charging solutions has become a promising and important area of research. One such class of solutions is based on optimization techniques such as mixed-integer programming. While being effective, these techniques require precise data and are computationally intensive. To counteract this, and with the rising popularity and efficacy of machine learning, researchers have applied supervised-learning based solutions to the problem of charging a single EV in a home environment (HEMS) [Huy et al., 2023]. However, this novel concept has been largely untested in environments with more than one EV. Hence, we apply supervised-learning (SL) to situations with multiple EVs and show that by using the MILP as an oracle, we can train models that perform reasonably well in multi-EV settings. We further analyze our models to understand their internal decision-making processes and identify key limitations. Based on these insights, we propose a decentralized SL approach and a reinforcement-learning approach, which are able to match or exceed SL performance.

All experiments were conducted using the EV-SIM simulation environment [Chau et al., 2000].


## Results
The following figure illustrates the performance of the proposed approaches in a multi-EV charging scenario, compared to the MILP and MPC baselines.


![alt text](image.png)

Overall, while the simple supervised-learning approach struggles to fully replicate the expert MILP policy, the decentralized SL and certain reinforcement-learning algorithms, particularly PPO and TRPO, demonstrate competitive performance, achieving high user satisfaction and strong profitability in multi-EV charging scenarios.

This highlights the potential of these approaches for real-world scenarios, where system knowledge is often incomplete and computational resources are limited.

## References

- Truong Hoang Bao Huy, Huy Truong Dinh, Dieu Ngoc Vo, & Daehee Kim. *Real-time energy scheduling for home energy management systems with an energy storage system and electric vehicle based on a supervised-learning-based strategy*. **Energy Conversion and Management**, Volume 292, 2023, 117340. https://doi.org/10.1016/j.enconman.2023.117340
- Chau, K.T. & Wong, Y.s & Chan, Ching. (2000). EVSIM — A PC-based Simulation Tool for an Electric Vehicle Technology Course. International Journal of Electrical Engineering Education. 37. 10.7227/IJEEE.37.2.6.
