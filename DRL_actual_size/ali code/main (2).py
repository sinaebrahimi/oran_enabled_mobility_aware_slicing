import numpy as np
from  scipy.stats import rayleigh
import matplotlib.pyplot as plt
#-------------------------------------------
from sac_torch import Agent
np.random.seed(1373)
#%%
x_lim=100
y_lim=100
number_HD=10
number_AP=4
number_subchannel=32
max_size_task=1e6
#$-------------------
beta_max=.9
beta_min=.1
#$-------------------
E_max=.1
P_max_AP=1
sigma_noise=10e-17
bandwith=200e3
#%%
##########################################################################################
class location_channel:
    def __init__(self):
        self.loc_AP=np.random.rand(2,number_AP)*x_lim
        self.accosiator_HD_AP=np.zeros([number_HD,number_AP])
        #------------------------------------------------------------------
        self.loc_HD=np.random.rand(2,number_HD)*x_lim
        #------------------------------------------------------------------
        self.H=np.zeros([number_AP,number_subchannel,number_HD])
        self.H_hat=np.zeros([number_AP,number_subchannel,number_HD])
        #------------------------------------------------------------------
        
    def _(self):
        for u in range(number_HD):
          self.mem_ap=[]
          self.xu = self.loc_HD[0,u]
          self.yu = self.loc_HD[1,u]
          for ap in range(number_AP):
              dd=(self.loc_AP[0,ap]-self.xu)**2 + (self.loc_AP[1,ap]-self.yu)**2
              self.mem_ap.append(dd)
              dd=dd**.5
              dd=dd**(-3)
              H_u=rayleigh.rvs(np.ones(number_subchannel))*dd
              H_hat_u=rayleigh.rvs(np.ones(number_subchannel))*dd
              self.H[ap,:,u]=H_u
              self.H_hat[ap,:,u]=H_hat_u
              
          self.mem_ap=np.array(self.mem_ap)
          temp_ap=np.argmin(self.mem_ap)
          self.accosiator_HD_AP[u,temp_ap]=1
          
        return self.H, self.H_hat ,self.accosiator_HD_AP
#%%##########################################################################################
class task_gen:
    def __init__(self,flag_gen):
        self.flag_gen=flag_gen
        self.mat_task=np.zeros(number_HD)
        
    def _(self):
        for u in range(number_HD):
            if self.flag_gen[u]==1:
                self.mat_task[u]=np.random.rand()*max_size_task
        return self.mat_task
##########################################################################################
class eng_gen:
    def __init__(self,mat_energy,alpha_PH):
        self.mat_energy=mat_energy
        self.alpha_PH=alpha_PH
    def _(self):
        for u in range(number_HD):
            self.mat_energy[u]=self.mat_energy[u]+np.sum(self.alpha_PH[:,:,u])
        return self.mat_energy
#%%##########################################################################################
class value_gen:
    def __init__(self,mat_valu,inher_valu,flag_gen,flag_have):
        self.coeff=.1
        self.mat_valu=mat_valu
        self.flag_gen=flag_gen
        self.flag_have=flag_have
        self.inher_valu=inher_valu
    def _decay_(self):
        for u in range(number_HD):
            if  self.flag_have[u]==1:
                self.mat_valu[u]=self.mat_valu[u]*(2.71**(-self.coeff))*self.inher_valu[u]
        return self.mat_valu
    
    def _generat_(self):
        for u in range(number_HD):
            if self.flag_gen[u]==1:
                self.mat_valu[u]=1*self.inher_valu[u]
        return self.mat_valu        
#%%##########################################################################################
class mapping:
    def __init__(self,action,mat_energy,accosiator_HD_AP):
        self.action=action
        self.mat_energy=mat_energy
        self.accosiator_HD_AP=accosiator_HD_AP
        #----------------------------------------------------
        self.e0=0
        #----------------------------------------------------
        self.e1=number_AP*number_subchannel*number_HD
        #----------------------------------------------------
        self.e2=self.e1+number_AP*number_subchannel*number_HD
        #----------------------------------------------------
        self.e3=self.e2+number_AP*number_subchannel*number_HD
        #----------------------------------------------------
        self.e4=self.e3+number_AP*number_subchannel*number_HD
        #----------------------------------------------------
        self.e5=self.e4 + number_AP*number_subchannel*number_HD
        #----------------------------------------------------
        self.e6=self.e5 + number_HD
        ################################################################
        
    def _rho_down(self):
        self.rho=np.zeros([number_AP,number_subchannel,number_HD])
        self.temp_action=self.action[self.e0:self.e1]
        self.temp_action_resh=np.reshape(self.temp_action,[number_AP,number_subchannel,number_HD])
        for k in range(number_subchannel):
            for ap in range(number_AP):
                list_u = np.where(self.accosiator_HD_AP[:,ap]==1)
                self.temp_rho=np.zeros([number_AP,number_subchannel,number_HD])
                self.temp_rho[ap,k,list_u]=1
                self.temp_rho_prod = self.temp_rho*self.temp_action_resh
                self.temp_u=np.argmax(self.temp_rho_prod[ap,k,:])
                self.rho[ap,k,self.temp_u]=1
                
        cnt_u=0
        done_u=0
        for u in range(number_HD):
            if np.sum(self.rho[:,:,u])>=1:
                cnt_u+=1
                
        if cnt_u==number_HD:
            done_u=1
                            
        return self.rho, done_u
        
    def _rho_up(self):
        self.rho_hat=np.zeros([number_AP,number_subchannel,number_HD])
        self.temp_action=self.action[self.e1:self.e2]
        self.temp_action_resh=np.reshape(self.temp_action,[number_AP,number_subchannel,number_HD])
        for k in range(number_subchannel):
            for ap in range(number_AP):
                list_u = np.where(self.accosiator_HD_AP[:,ap]==1)
                self.temp_rho=np.zeros([number_AP,number_subchannel,number_HD])
                self.temp_rho[ap,k,list_u]=1
                self.temp_rho_prod = self.temp_rho*self.temp_action_resh
                self.temp_u=np.argmax(self.temp_rho_prod[ap,k,:])
                self.rho_hat[ap,k,self.temp_u]=1
                
        cnt_u=0
        done_u=0
        for u in range(number_HD):
            if np.sum(self.rho_hat[:,:,u])>=1:
                cnt_u+=1
                
        if cnt_u==number_HD:
            done_u=1
                            
        return self.rho_hat, done_u
    
    
    def _power_down(self,rho):
        self.mat_remain_power=np.ones(number_AP)*P_max_AP
        self.scale=.0005
        self.p=np.zeros([number_AP,number_subchannel,number_HD])
        self.temp_action=self.action[self.e2:self.e3]
        self.temp_action_resh=np.reshape(self.temp_action,[number_AP,number_subchannel,number_HD])
        self.temp_action_resh_new=((self.temp_action_resh+1)/2)+.00005
        cnt_u=0
        done_u=0
        for u in range(number_HD):
            for ap in range(number_AP):
                if self.accosiator_HD_AP[u,ap]==1:
                    for k in range(number_subchannel):
                        if rho[ap,k,u]==1:
                            if self.mat_remain_power[ap]-self.scale*P_max_AP>0:
                                if self.scale*self.temp_action_resh_new[ap,k,u]>0:
                                    self.p[ap,k,u]=self.scale*self.temp_action_resh_new[ap,k,u]
                                    self.mat_remain_power[ap]=self.mat_remain_power[ap]-self.scale*self.temp_action_resh_new[ap,k,u]
                                
            if np.sum(self.p[:,:,u])>0:
                cnt_u+=1
                
        if cnt_u==number_HD:
            done_u=1

        return self.p, done_u
    
    def _power_up(self, rho_hat):
        self.temp_action=self.action[self.e3:self.e4]
        self.temp_action_resh=np.reshape(self.temp_action,[number_AP,number_subchannel,number_HD])
        self.temp_action_resh_new=((self.temp_action_resh+1)/2 +.00005)
        self.scale=0.0005
        self.p_hat=np.zeros([number_AP,number_subchannel,number_HD])
        cnt_u=0
        done_u=0
        for u in range(number_HD):
            for ap in range(number_AP):
                if self.accosiator_HD_AP[u,ap]==1:
                    for k in range(number_subchannel):
                        if rho_hat[ap,k,u]==1:
                            if self.mat_energy[u]-self.scale*self.temp_action_resh_new[ap,k,u]>0:
                                if self.scale*self.temp_action_resh_new[ap,k,u]>0:
                                    self.p_hat[ap,k,u]=self.scale*self.temp_action_resh_new[ap,k,u]
                                    self.mat_energy[u]=self.mat_energy[u]-self.scale*self.temp_action_resh_new[ap,k,u]
                                 
            if np.sum(self.p_hat[:,:,u])>0:
                cnt_u+=1
                
        if cnt_u==number_HD:
            done_u=1
            
        return self.p_hat, self.mat_energy, done_u
    
    def _alpha(self):
        self.temp_action=self.action[self.e4:self.e5]
        self.temp_action=(self.temp_action+1)/2
        self.temp_action_resh=np.reshape(self.temp_action,[number_AP,number_subchannel,number_HD])
        self.alpha=np.zeros([number_AP,number_subchannel,number_HD])
        for u in range(number_HD):
            for ap in range(number_AP):
                if self.accosiator_HD_AP[u,ap]==1:
                    for k in range(number_subchannel):
                        self.alpha[ap,k,u]=self.temp_action_resh[ap,k,u]
                else:
                     self.alpha[ap,:,u]=0                            
        return self.alpha
    
    def _beta(self):
        self.temp_action=self.action[self.e4:self.e5]
        self.temp_action=(self.temp_action+1)/2
        self.temp_action=np.clip(self.temp_action, beta_min, beta_max)
        self.beta=self.temp_action
        return self.beta
#%%
class rate_cal:
    def __init__(self,rho,rho_hat,p,p_hat,H,H_hat):
        self.rho=rho
        self.rho_hat=rho_hat
        self.p=p
        self.p_hat=p_hat
        self.H=H
        self.H_hat=H_hat
    def __downlink__(self):
        self.min_rate_down=1
        self.done_total_down = 0
        self.mat_done_down = np.zeros([number_HD])
        self.mat_rate_down = np.zeros([number_AP,number_subchannel,number_HD])
        for u in range(number_HD):
            self.comul=0
            for ap in range(number_AP):
                for k in range(number_subchannel):
                    if self.rho[ap,k,u]>0:
                        ph = self.p[ap,k,u]*self.H[ap,k,u]*self.rho[ap,k,u]
                        I_intr=rate_cal.__inter__(self, ap, k, u, self.H, self.p, self.rho)
                        SINR = ph/(sigma_noise+I_intr) 
                        if SINR>0:
                            self.mat_rate_down[ap,k,u]=bandwith*np.log2(1+SINR)
                            self.comul+=bandwith*np.log2(1+SINR)
                        
            if self.comul>=self.min_rate_down:
                self.mat_done_down[u]=1
        if np.sum(self.mat_done_down)==number_HD:
            self.done_total_down=1
            
        return self.mat_rate_down, self.done_total_down, self.mat_done_down
#---------------------------------------------------------
    def __uplink__(self):
        self.min_rate_up=1
        self.done_total_up = 0
        self.mat_done_up = np.zeros([number_HD])    
        self.mat_rate_up=np.zeros([number_AP,number_subchannel,number_HD])
        for u in range(number_HD):
            self.comul=0
            for ap in range(number_AP):
                for k in range(number_subchannel):
                    if self.rho_hat[ap,k,u]>0:
                        ph = self.p_hat[ap,k,u]*self.H_hat[ap,k,u]
                        I_intr=rate_cal.__inter__(self, ap, k, u, self.H_hat, self.p_hat, self.rho_hat)
                        SINR = ph/(sigma_noise+I_intr) 
                        if SINR>0:
                            self.mat_rate_up[ap,k,u]=bandwith*np.log2(1+SINR)
                            self.comul+=bandwith*np.log2(1+SINR)
                    
            if self.comul>=self.min_rate_up:
                self.mat_done_up[u]=1
        if np.sum(self.mat_done_up)==number_HD:
            self.done_total_up=1
        return self.mat_rate_up, self.done_total_up, self.mat_done_up
    
    def __inter__(self,ap,k,u,H,p,rho):
        self.I_intr=0
        for uu in range(number_HD):
            if uu != u:
                for app in range(number_AP):
                    if app != ap:
                        self.I_intr+=H[app,k,uu]*p[app,k,uu]*rho[app,k,uu]
        return  self.I_intr
#%%---------------------------------------------------------------------
class state_cal:
    def __init__(self,H,H_hat,mat_energy):
        self.H=H
        self.H_hat=H_hat
        self.mat_energy=mat_energy
        self.state_size = 2*self.H.size + self.mat_energy.size 
        self.state=np.zeros(self.state_size)
        
    def _(self):
        self.H_resh=np.reshape(self.H,[1,self.H.size])
        self.H_hat_resh=np.reshape(self.H_hat,[1,self.H_hat.size])
        self.mat_energy_resh=np.reshape(self.mat_energy,[1,self.mat_energy.size])
        self.state[0:self.H.size] = self.H_resh
        self.state[self.H.size:2*self.H.size] = self.H_hat_resh
        self.state[2*self.H.size::]=self.mat_energy_resh
        self.state=self.state/max(self.state)
        return self.state
#%%---------------------------------------------------------------------
class __run__:
    def __init__(self,T):
        self.T=T
        self.mat_Reward=np.zeros([self.T])
        self.mat_Val_time=np.zeros([self.T,number_HD])
        self.mat_power_consumed=np.zeros([self.T])
        alpha=.0000001
        beta=.00000001
        self.n_actions=5*number_AP*number_subchannel*number_HD + number_HD
        input_dims=2*number_AP*number_subchannel*number_HD + number_HD
        self.agent=Agent(alpha, beta, self.n_actions, input_dims)
        self.mat_energy=E_max*np.ones(number_HD)
        self.var=1
        self.decay_var=.9999
        self.LC=location_channel()
        self.inher_valu = np.random.rand(number_HD)+.5
        self.ep_rewardall=[]
        self.mem_critc=[]
        self.mem_act=[]
    def __(self):
        for t in range(self.T):
            print(t)
            w=50
            if t>0:
                if np.mod(t,w)==0:
                    aaa=len(self.ep_rewardall)
                    mean_ep_rewardall=[]
                    for i in range(aaa-w) :
                        temp_value= np.sum(self.ep_rewardall[i:-aaa + w+i])/w
                        mean_ep_rewardall.append(temp_value) 
                    plt.plot(mean_ep_rewardall, label='Mean of Reward')
                    plt.xlabel("Episode")
                    plt.ylabel("Mean of Reward")
                    plt.legend()
                    plt.show()    
                    #------------------------------------------
                    # mean_ep_c=[]
                    # aaa=len(self.mem_critc)
                    # mean_ep_rewardall=[]
                    # for i in range(aaa-w) :
                    #     temp_value= np.sum(self.mem_critc[i:-aaa + w+i])/w
                    #     mean_ep_rewardall.append(temp_value) 
                    # plt.plot(mean_ep_rewardall, label='Mean of Loss')
                    # plt.xlabel("Episode")
                    # plt.ylabel("Mean of loss_crit")
                    # plt.legend()
                    # plt.show()    
                    # #--------------------------------------
                    # mean_ep_a=[]
                    # aaa=len(self.mem_act)
                    # mean_ep_rewardall=[]
                    # for i in range(aaa-w) :
                    #     temp_value= np.sum(self.mem_act[i:-aaa + w+i])/w
                    #     mean_ep_rewardall.append(temp_value) 
                    # plt.plot(mean_ep_rewardall, label='Mean of Loss')
                    # plt.xlabel("Episode")
                    # plt.ylabel("Mean of loss_act")
                    # plt.legend()
                    # plt.show()  
                    
                    
            self.reward=-1
            self.H, self.H_hat, self.accosiator_HD_AP = self.LC._()
            SC=state_cal(self.H, self.H_hat, self.mat_energy)
            self.state=SC._()
            #-----------------------------------------
            self.action = self.agent.choose_action(self.state)
            #--------------------------
            self.action_pure = np.copy(self.action)
            #--------------------------
            self.var=self.var*self.decay_var
            self.noise=np.random.randn(self.n_actions)
            # self.noise=self.noise*self.var          
            self.action= self.action + self.noise
            self.action = np.clip(self.action,-1,1)
            #-----------------------------------------
            MA=mapping(self.action, self.mat_energy, self.accosiator_HD_AP)
            self.rho, done_u1 = MA._rho_down()
            self.rho_hat, done_u2=MA._rho_up()
            self.p, done_u3=MA._power_down(self.rho)
            self.p_hat, self.mat_energy_new, done_u4=MA._power_up(self.rho_hat)
            self.alpha=MA._alpha()
            self.beta=MA._beta()
            #--------------------------------------------
            if done_u1==1:
                if done_u2==1:
                    if done_u3==1:
                        if done_u4==1:
                            print("done_prime")
                            self.mat_energy=self.mat_energy_new
                            self.alpha_PH=self.rho*self.H
                            EG=eng_gen(self.mat_energy, self.alpha_PH)      
                            self.mat_energy=EG._()
            #--------------------------------------
            RC=rate_cal(self.rho, self.rho_hat, self.p, self.p_hat, self.H, self.H_hat)
            self.mat_rate_down, self.done_total_down, self.mat_done_down = RC.__downlink__()
            self.mat_rate_up, self.done_total_up, self.mat_done_up = RC.__uplink__()
            
            #==================================================================
            if t==0:
                self.mat_valu=np.ones([number_HD])
                self.flag_have=np.ones([number_HD])
                self.flag_gen=np.ones([number_HD])
            else:
                if np.sum(self.flag_have)==0:
                    if np.mod(t, 10)==0:
                        self.flag_gen=np.ones([number_HD])
                        self.flag_have=np.ones([number_HD])

            self.VG=value_gen(self.mat_valu, self.inher_valu, self.flag_gen, self.flag_have)
            if self.done_total_down==1:
                if self.done_total_up==1:
                    print("done_second")
                    self.flag_have=np.zeros([number_HD])
                    self.flag_gen=np.zeros([number_HD])
                    self.mat_power_consumed[t]=(np.sum(self.p) + np.sum(self.p_hat))
                    self.reward =  (np.sum(self.mat_rate_down)+np.sum(self.mat_rate_up))/(10e6)  #10-1*((np.sum(self.p) + np.sum(self.p_hat)))
                else:
                    self.reward -= 5
                    self.mat_valu=self.VG._decay_()
                    
            # print(self.mat_energy)
            # print("reward:::",self.reward)
            # print(self.action)
            self.flag_gen=np.zeros([number_HD])
            SC=state_cal(self.H, self.H_hat, self.mat_energy)
            self.state_new=SC._()
            
            self.agent.memorize(self.state, self.action_pure, self.reward, self.state_new)
            self.agent.replay()
            # self.mem_critc.append(loss_c)
            # self.mem_act.append(loss_a)
            #--------------------------------------
            self.mat_Reward[t] = self.reward
            self.mat_Val_time[t,:]= self.mat_valu
            self.ep_rewardall.append(self.reward)
            
        return  self.mat_Reward,  self.mat_Val_time, self.mat_power_consumed
#%%---------------------------------------------------------------
T=10000
RUN=__run__(T)
mat_Reward,  mat_Val_time, mat_power_consumed=RUN.__()
 
                    
                

