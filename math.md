At each 50 Hz control step $t$, the reward $R_t$ is computed as:

$$R_t = \underbrace{R_{\text{closure}}}_{\text{Approach Phase}} + \underbrace{R_{\text{basket}}}_{\text{Engagement Phase}} + \underbrace{R_{\text{velocity\_matching}}}_{\text{Stability}} - \underbrace{P_{\text{boundary}}}_{\text{FOV Soft Barrier}} - \underbrace{P_{\text{jerk}}}_{\text{Smoothness}}$$



$$R_{\text{closure}} = - w_{\text{close}} \cdot \tanh\left( \frac{|d_t - d^*|}{d_{\text{scale}}} \right)$$