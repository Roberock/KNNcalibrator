import numpy as np

def forward_propagation(pdf_theta, simulator, xi, n_samples):
    """
    Forward propagation p(y|xi)

    pdf_theta : object with method sample(n)
    simulator : function simulator(theta, xi) -> y
    xi        : design / decision variable
    n_samples : Monte Carlo samples
    """

    # sample parameters
    try:
        samples = pdf_theta.sample(n_samples)   # shape (n_samples, d_theta)
    except:
        try:
            samples = pdf_theta # passed as samples
        except:
            print('pass pdf object with .sample() method or pass samples to be propagated directly')
    # propagate through simulator
    y = np.array([simulator(theta, xi) for theta in samples])
    pdf_y = np.ravel(y)                                # returned as samples of p(y|xi)
    return pdf_y, {"theta": samples, "y":y}
