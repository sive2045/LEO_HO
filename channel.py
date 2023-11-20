import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt


def calculate_shadowed_rice_fading_gain(elevation_angle, debugging=False):
    # elevation angel [degree]
    
    # parameters
    # A. Abdi, W. C. Lau, M. -. Alouini and M. Kaveh, “A new simple model for land mobile satellite channels: First- and second-order statistics,” 
    # IEEE Wireless Commun. Lett., vol. 2, no. 3, pp. 519-528, May 2003
    # average power of multi path
    b = (-4.7943 * 10**(-8) * elevation_angle**(3) + 5.5784 * 10**(-6) * elevation_angle**(2) 
         -2.1344 * 10**(-4) * elevation_angle + 3.2710 * 10**(-2))
    # LoS componets
    omega = (
        1.4428 * 10**(-5) * elevation_angle**(3) -2.3798 * 10**(-3) * elevation_angle**(2)
        +1.2702 * 10**(-1) * elevation_angle -1.4864
    )
    # Nakagami-m fading
    m = (
        6.3739 * 10**(-5) * elevation_angle**(3) +5.8533 * 10**(-4) * elevation_angle**(2)
        -1.5973 * 10**(-1) * elevation_angle + 3.5156
    )

    # Shadowed Rice fading model PDF
    x = np.linspace(0,4,1000)
    PDF = 1/(2*b) * (((2*b*m)/(2*b*m+omega))**(m)) * np.exp(-x/(2*b)) * sc.hyp1f1(m,1,omega*x/(2*b*(2*b*m+omega)))

    # plot PDF (for debugging)
    if debugging:
        plt.plot(x,PDF)
        plt.title(f'elevation angle: {elevation_angle} PDF')
        plt.show()

    # Sampling from PDF
    idx = np.random.randint(0,1000)
    channel = PDF[idx]

    return channel

def calculate_FSPL(dist,f):
    FSPL = 4*np.pi*(dist)*f/(3*10**(8))

def data_rate(channel_gain):
    BW = 2_000_000
    load = 3
    N = np.random.randn(1)[0]**2

    rate = BW/load * np.log2(1 + (40*0.9*1000*channel_gain /(3*40*0.9*1000*channel_gain+ 10**(-3.5))))

    return rate

if __name__ == '__main__':
    # channel_gain = calcuate_channel_gain(90, 550)
    # rate = calculate_data_rate(channel_gain, 40,2)
    # print(rate)
    channel_gain = calculate_shadowed_rice_fading_gain(85)
    print(channel_gain)
    rate = data_rate(channel_gain)
    print(rate)