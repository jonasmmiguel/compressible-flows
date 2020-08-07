from src.workbench import isentropic as isen, nshock, fanno, rayleigh
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')  # visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots  # multi-axes plots
from matplotlib.colors import to_rgba
import plotly.io as pio  # exporting plotly plots to HTML
from chart_studio import tools as tls  # get iframe
import chart_studio.plotly as py  # get URL


def set_rgba_color(css_code='MidnightBlue', alpha=0.8):
    return 'rgba' + str(to_rgba(css_code, alpha=alpha))


def plot_isentropic():
    M_space = np.linspace(1E-2, 6.0, int(1E+03))
    r_A_Astar = [isen('A', M=M) for M in M_space]
    r_T_Tt = [isen('T', M=M) for M in M_space]
    r_p_pt = [isen('p', M=M) for M in M_space]

    # Create figure, with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]],
                        subplot_titles=['<b>Isentropic Flow Functions</b>'],
                        )

    # Create traces
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_A_Astar,
                             mode='lines',
                             name=r'$A/A^*$',
                             line={'color': set_rgba_color('LightSlateGrey', alpha=0.5)}
                             ),
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_T_Tt,
                             mode='lines',
                             name=r'$T/T_t$',
                             line={'color': set_rgba_color('DarkOrange', alpha=0.5)}
                             ),
                  secondary_y=True
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_p_pt,
                             mode='lines',
                             name=r'$p/p_t$',
                             line={'color': set_rgba_color('Green', alpha=0.5)}
                             ),
                  secondary_y=True
                  )

    # Set x-axis title
    fig.update_xaxes(title_text=r'$M$')

    # Set y-axes titles
    fig.update_yaxes(title_text=r'$A/A^*$')
    fig.update_yaxes(title_text=r'$T/T_t, p/p_t$',
                     overlaying='y1',
                     range=[0, np.inf],
                     secondary_y=True)

    # Configure layout
    fig.update_layout(legend={'x': 0.75, 'y': 0.95})

    # save plot as HTML
    # pio.write_html(fig, file='isentropic.html', auto_open=False)
    #
    # host chart in Plotly Chart Studio
    py.plot(fig, filename='isentropic', auto_open=True)
    # fig.show()


def plot_nshock():
    Ms_space = np.linspace(1+1E-2, 6.0, int(1E+03))
    Msl = [nshock('Msl', Ms=Ms) for Ms in Ms_space]
    r_psl_ps = [nshock('p', Ms=Ms) for Ms in Ms_space]
    r_ptsl_pts = [nshock('pt', Ms=Ms) for Ms in Ms_space]

    # Create figure, with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]],
                        subplot_titles=['<b>Normal Shock Functions</b>'],
                        )

    # Create traces
    fig.add_trace(go.Scatter(x=Ms_space,
                             y=Msl,
                             mode='lines',
                             name=r'$M_{s^\prime}$',
                             line={'color': set_rgba_color('MediumBlue', alpha=0.5)}
                             ),
                  )
    fig.add_trace(go.Scatter(x=Ms_space,
                             y=r_psl_ps,
                             mode='lines',
                             name=r'$p_{s^\prime}/p_{s}$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.5))}
                             ),
                  secondary_y=True
                  )
    fig.add_trace(go.Scatter(x=Ms_space,
                             y=r_ptsl_pts,
                             mode='lines',
                             name=r'$p_{t, s^\prime}/p_{t,s}$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.8))}
                             ),
                  )

    # Set x-axis title
    fig.update_xaxes(title_text=r'$M_{s}$')

    # Set y-axes titles
    fig.update_yaxes(title_text=r'$M_{s^\prime}, p_{t, s^\prime}/p_{t,s}$', secondary_y=False)
    fig.update_yaxes(title_text=r'$p_{s^\prime}/p_{s}$',
                     overlaying='y1',
                     range=[0, np.inf],
                     secondary_y=True)

    # Configure layout
    fig.update_layout(legend={'x': 0.75, 'y': 0.95})

    # save plot as HTML
    # pio.write_html(fig, file='nshock.html', auto_open=True, include_mathjax='cdn')

    # get iframe
    # plot_url = py.plot(fig, filename='nshock.html')
    # py.iplot(fig, filename='nshock.html', sharing='public')
    py.plot(fig, filename='nshock', auto_open=True)
    #fig.show()


def plot_fanno():
    M_space = np.linspace(1E-2, 6.0, int(1E+03))

    r_T_Tstar = [fanno('T', M=M) for M in M_space]
    r_p_pstar = [fanno('p', M=M) for M in M_space]
    r_p_pt = [fanno('pt', M=M) for M in M_space]
    fldmax = [fanno('fld', M=M) for M in M_space]

    # Create figure, with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]],
                        subplot_titles=['<b>Fanno Flow Functions</b>'],
                        )

    # Create traces
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_T_Tstar,
                             mode='lines',
                             name=r'$T/T^*$',
                             line={'color': 'rgba' + str(to_rgba('DarkOrange', alpha=0.5))}
                             )
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_p_pstar,
                             mode='lines',
                             name=r'$p/p^*$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.5))}
                             ),
                  secondary_y=True
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_p_pt,
                             mode='lines',
                             name=r'$p/p_t$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.8))}
                             ),
                  secondary_y=True
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=fldmax,
                             mode='lines',
                             name=r'$fL_{max}/D$',
                             line={'color': 'rgba' + str(to_rgba('LightSlateGrey', alpha=0.8))}
                             )
                  )

    # Set x-axis title
    fig.update_xaxes(title_text=r'$M$')

    # Set y-axes titles
    fig.update_yaxes(title_text=r'$fL_{max}/D,\ T/T^*$', secondary_y=False)
    fig.update_yaxes(title_text='$p/p_t, p/p^*$', secondary_y=True)

    # Configure layout
    fig.update_layout(yaxis_type="log",
                      legend={'x': 0.75, 'y': 0.95}
                      )

    # save plot as HTML
    # pio.write_html(fig, file='fanno.html', auto_open=True)

    # host chart in Plotly Chart Studio
    # py.iplot(fig, filename='fanno.html', sharing='public')
    py.plot(fig, filename='fanno', auto_open=True)

    # fig.show()


def plot_rayleigh():
    M_space = np.linspace(1E-2, 6.0, int(1E+03))

    r_T_Tstar = [rayleigh('T', M=M) for M in M_space]
    r_Tt_Ttstar = [rayleigh('Tt', M=M) for M in M_space]
    r_p_pstar = [rayleigh('p', M=M) for M in M_space]
    r_pt_ptstar = [rayleigh('pt', M=M) for M in M_space]

    # Create figure, with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]],
                        subplot_titles=['<b>Rayleigh Flow Functions</b>'],
                        )

    # Create traces
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_T_Tstar,
                             mode='lines',
                             name=r'$T/T^*$',
                             line={'color': 'rgba' + str(to_rgba('DarkOrange', alpha=0.5))}
                             ),
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_Tt_Ttstar,
                             mode='lines',
                             name=r'$T_t/T_t^*$',
                             line={'color': 'rgba' + str(to_rgba('DarkOrange', alpha=0.8))}
                             ),
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_p_pstar,
                             mode='lines',
                             name=r'$p/p^*$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.5))}
                             ),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_pt_ptstar,
                             mode='lines',
                             name=r'$p_t/p_t^*$',
                             line={'color': 'rgba' + str(to_rgba('Green', alpha=0.8))}
                             ),
                  secondary_y=True)

    # Set x-axis title
    fig.update_xaxes(title_text=r'$ M $')

    # Set y-axes titles
    fig.update_yaxes(title_text=r'$T/T^*,\  T_t/T_t^*$', secondary_y=False)
    fig.update_yaxes(title_text=r'$p/p^*,\  p_t/p_t^*$', secondary_y=True)

    # Configure layout
    fig.update_layout(legend={'x': 0.75, 'y': 0.95})

    # save plot as HTML
    # pio.write_html(fig, file='rayleigh.html', auto_open=True)

    # host chart in Plotly Chart Studio
    # py.iplot(fig, filename='rayleigh.html', sharing='public')
    py.plot(fig, filename='rayleigh', auto_open=True)
    # fig.show()


def plot_all():
    plot_isentropic()
    plot_nshock()
    plot_fanno()
    plot_rayleigh()


if __name__ == '__main__':


    print('ready!')
