from src.workbench import isentropic as isen, nshock, fanno, rayleigh
import numpy as np
import matplotlib.pyplot as plt; plt.style.use('ggplot')  # visualization
import plotly.graph_objects as go
from plotly.subplots import make_subplots

if __name__ == '__main__':

    # '#1f77b4',  // muted blue
    # '#ff7f0e',  // safety orange
    # '#2ca02c',  // cooked asparagus green
    # '#d62728',  // brick red
    # '#9467bd',  // muted purple
    # '#8c564b',  // chestnut brown
    # '#e377c2',  // raspberry yogurt pink
    # '#7f7f7f',  // middle gray
    # '#bcbd22',  // curry yellow-green
    # '#17becf'   // blue-teal

    # # ==========================================
    # # Isentropic
    # # ==========================================
    # M_space = np.linspace(1E-2, 6.0, int(1E+03))
    # r_A_Astar = [isen('A', M=M) for M in M_space]
    # r_T_Tt = [isen('T', M=M) for M in M_space]
    # r_p_pt = [isen('p', M=M) for M in M_space]
    #
    # # Create figure, with secondary y-axis
    # fig = make_subplots(specs=[[{'secondary_y': True}]],
    #                     subplot_titles=['<b>Isentropic Flow Functions</b>'],
    #                     )
    #
    # # Create traces
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_A_Astar,
    #                          mode='lines',
    #                          name='A/A*'),
    #               )
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_T_Tt,
    #                          mode='lines',
    #                          name='T/Tt'),
    #               secondary_y=True)
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_p_pt,
    #                          mode='lines',
    #                          name='p/pt'),
    #               secondary_y=True)
    #
    # # Set x-axis title
    # fig.update_xaxes(title_text='M [-]')
    #
    # # Set y-axes titles
    # fig.update_yaxes(title_text='A/A*', secondary_y=False)
    # fig.update_yaxes(title_text='T/Tt, p/pt', secondary_y=True)
    #
    # fig.show()

    # # ==========================================
    # # Normal Shock
    # # ==========================================
    # Ms_space = np.linspace(1+1E-2, 6.0, int(1E+03))
    # Msl = [nshock('Msl', Ms=Ms) for Ms in Ms_space]
    # r_psl_ps = [nshock('p', Ms=Ms) for Ms in Ms_space]
    # r_ptsl_pts = [nshock('pt', Ms=Ms) for Ms in Ms_space]
    #
    # # Create figure, with secondary y-axis
    # fig = make_subplots(specs=[[{'secondary_y': True}]],
    #                     subplot_titles=['<b>Normal Shock Functions</b>'],
    #                     )
    #
    # # Create traces
    # fig.add_trace(go.Scatter(x=Ms_space,
    #                          y=Msl,
    #                          mode='lines',
    #                          name='Msl'),
    #               )
    # fig.add_trace(go.Scatter(x=Ms_space,
    #                          y=r_psl_ps,
    #                          mode='lines',
    #                          name='psl/ps'),
    #               secondary_y=True)
    # fig.add_trace(go.Scatter(x=Ms_space,
    #                          y=r_ptsl_pts,
    #                          mode='lines',
    #                          name='ptsl/pts'),)
    #
    # # Set x-axis title
    # fig.update_xaxes(title_text='M [-]')
    #
    # # Set y-axes titles
    # fig.update_yaxes(title_text='Msl, ptsl/pts', secondary_y=False)
    # fig.update_yaxes(title_text='psl/ps', secondary_y=True)
    #
    # fig.show()

    # # ==========================================
    # # Fanno
    # # ==========================================
    # M_space = np.linspace(1E-2, 6.0, int(1E+03))
    #
    # r_T_Tstar = [fanno('T', M=M) for M in M_space]
    # r_p_pstar = [fanno('p', M=M) for M in M_space]
    # r_p_pt = [fanno('pt', M=M) for M in M_space]
    # fldmax = [fanno('fld', M=M) for M in M_space]
    #
    # # Create figure, with secondary y-axis
    # fig = make_subplots(specs=[[{'secondary_y': True}]],
    #                     subplot_titles=['<b>Fanno Flow Functions</b>'],
    #                     )
    #
    # # Create traces
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_T_Tstar,
    #                          mode='lines',
    #                          name='T/T*'))
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_p_pstar,
    #                          mode='lines',
    #                          name='p/p*'),
    #               secondary_y=True)
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=r_p_pt,
    #                          mode='lines',
    #                          name='p/pt'),
    #               secondary_y=True)
    # fig.add_trace(go.Scatter(x=M_space,
    #                          y=fldmax,
    #                          mode='lines',
    #                          name='fLmax/d'))
    #
    # # Set x-axis title
    # fig.update_xaxes(title_text='M [-]')
    #
    # # Set y-axes titles
    # fig.update_yaxes(title_text='fLmax/d, T/T*', secondary_y=False)
    # fig.update_yaxes(title_text='p/pt, p/p*', secondary_y=True)
    #
    # # Set y-axes limits
    # #fig.update_layout(yaxis=dict(range=[-20, 1E+02]))
    # fig.update_layout(yaxis_type="log")
    # fig.show()

    # ==========================================
    # Rayleigh
    # ==========================================
    M_space = np.linspace(1E-2, 6.0, int(1E+03))

    r_T_Tstar = [rayleigh('T', M=M) for M in M_space]
    r_Tt_Ttstar = [rayleigh('Tt', M=M) for M in M_space]
    r_p_pstar = [fanno('p', M=M) for M in M_space]
    r_pt_ptstar = [fanno('pt', M=M) for M in M_space]

    # Create figure, with secondary y-axis
    fig = make_subplots(specs=[[{'secondary_y': True}]],
                        subplot_titles=['<b>Rayleigh Flow Functions</b>'],
                        )

    # Create traces
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_T_Tstar,
                             mode='lines',
                             name=r'$T/T^*$',
                             line={'color': '#ff7f0e'}),
                  )
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_Tt_Ttstar,
                             mode='lines',
                             name=r'$T_t/T_t^*$'))
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_p_pstar,
                             mode='lines',
                             name=r'$p/p^*$',),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=M_space,
                             y=r_pt_ptstar,
                             mode='lines',
                             name=r'$p_t/p_t^*$',),
                  secondary_y=True)

    # Set x-axis title
    fig.update_xaxes(title_text=r'$ M $')

    # Set y-axes titles
    fig.update_yaxes(title_text=r'$T/T^*,\  T_t/T_t^*$', secondary_y=False)
    fig.update_yaxes(title_text=r'$p/p^*,\  p_t/p_t^*$', secondary_y=True)

    # Set y-axes limits
    #fig.update_layout(yaxis=dict(range=[-20, 1E+02]))
    # fig.update_layout(yaxis_type="log")
    fig.show()

    print('ready!')
