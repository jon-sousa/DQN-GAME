from random import randint
import pygame
from pygame import font
from pygame import display
from pygame.image import load
from pygame.transform import scale, rotate
from pygame.sprite import Sprite, Group, GroupSingle, groupcollide
from pygame import event
from pygame.locals import QUIT, KEYUP, K_SPACE
from pygame.time import Clock
from math import cos, sin, pi, ceil
from random import randint, random
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd

EPISODES = 1000
LEARNING_RATE = 0.0000001
MEM_SIZE = 100000
BATCH_SIZE = 100
GAMMA = 0.5
EXPLORATION_MAX = 1.0
EXPLORATION_DECAY = 0.0001
EXPLORATION_MIN = 0.01

FC1_DIMS = 365
FC2_DIMS = 365
DEVICE = torch.device("cpu")


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, 365)
        self.to(DEVICE)
    
    def forward(self, x):
        x = torch.relu (self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class ReplayBuffer:
    def __init__(self):
        self.mem_count = 0
        
        self.states = np.zeros((MEM_SIZE, 1),dtype=np.float32)
        self.actions = np.zeros(MEM_SIZE, dtype=np.int64) 
        self.rewards = np.zeros(MEM_SIZE, dtype=np.float32)
        self.states_ = np.zeros((MEM_SIZE, 1),dtype=np.float32)
        self.dones = np.zeros(MEM_SIZE, dtype=bool)

    
    def add(self, state, action, reward, state_, done):
        mem_index = self.mem_count % MEM_SIZE
        
        self.states[mem_index]  = state
        self.actions[mem_index] = action
        self.rewards[mem_index] = reward
        self.states_[mem_index] = state_
        self.dones[mem_index] =  1 - done

        self.mem_count += 1
    
    def sample(self):
        MEM_MAX = min(self.mem_count, MEM_SIZE)
        batch_indices = np.random.choice(MEM_MAX, BATCH_SIZE, replace=True)
        
        states  = self.states[batch_indices]
        actions = self.actions[batch_indices]
        rewards = self.rewards[batch_indices]
        states_ = self.states_[batch_indices]
        dones   = self.dones[batch_indices]

        return states, actions, rewards, states_, dones

class DQN_Solver:
    def __init__(self):
        self.memory = ReplayBuffer()
        self.exploration_rate = EXPLORATION_MAX
        self.q_network = Network()
        self.target_network = Network()  
        self.update_target_network()
        self.step = 0 
        self.fixed_q = torch.tensor(np.ones(100)*100, dtype=torch.float32).to(DEVICE)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=LEARNING_RATE)     

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def choose_action(self, observation):
        aleatorio = float(randint(1, 99)) / 100
        if aleatorio < self.exploration_rate:
            action = randint(0, 364)
        else: 
            state = torch.tensor(observation).float().detach()
            state = state.to(DEVICE)
            state = state.unsqueeze(0)
            q_values = self.q_network(state)
            action = torch.argmax(q_values).item()

        velocidade = (int((action)/73) + 4)  
        angulo = int((action+1) % 73)
        
        if angulo == 0:
            angulo = 88

        self.exploration_rate -= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)

        return velocidade, angulo, action
    
    def learn(self):
        if self.memory.mem_count < 5000:
            return

        if self.step % 500 == 0:
            self.update_target_network()
        
        states, actions, rewards, states_, dones = self.memory.sample()
        states = torch.tensor(states , dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(actions, dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(DEVICE)
        states_ = torch.tensor(states_, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(dones, dtype=torch.bool).to(DEVICE)
        batch_indices = np.arange(BATCH_SIZE, dtype=np.int64)

        # next_q_values = self.target_network(states_)
        next_q_values = self.q_network(states_)
        predicted_value_of_future = torch.max(next_q_values, dim=1)[0]

        
        q_values = self.q_network(states)
        predicted_value_of_now = q_values[batch_indices, actions]
        q_target = rewards + GAMMA * predicted_value_of_future * dones

        loss = nn.functional.mse_loss(predicted_value_of_now, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.step += 1

    def returning_epsilon(self):
        return self.exploration_rate



pygame.init()

tamanho = 800, 600
fonte = font.SysFont('comicsans', 50)
fonte_perdeu = font.SysFont('comicsans', 300)

superficie = display.set_mode(
    size=tamanho,
)
display.set_caption(
    'Batalha Naval'
)

fundo = scale(
    load('images/fundo.png'),
    tamanho
)

class Mar(Sprite):
    def __init__(self, superficie):
        super().__init__()

        self.image = scale(load('images/mar.png'), (800,5))
        self.rect = self.image.get_rect()
        self.rect.x = 0
        self.rect.y = 540

class Navio(Sprite):
    def __init__(self):
        super().__init__()

        self.image = scale(load('images/navio2.png'), (100,50))
        self.rect = self.image.get_rect()
        self.rect.x = 30
        self.rect.y = 493
        self.velocidade = 2

    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            if self.rect.x > 0:
                self.rect.x -= self.velocidade

        if keys[pygame.K_RIGHT]:
            if self.rect.x < 150:
                self.rect.x += self.velocidade

class Navio_Inimigo(Sprite):
    def __init__(self, x):
        super().__init__()

        self.image = scale(load('images/navio-inimigo.png'), (100,50))
        self.rect = self.image.get_rect()
        self.rect.x = x
        self.rect.y = 490

    def update(self):
        pass

class Canhao(Sprite):
    def __init__(self):
        super().__init__()

        self.image = scale(load('images/canhao.png'), (15,30))
        self.rect = self.image.get_rect()
        self.angulo_da_mira = 0
        self.angulo = 90
        self.rect.x = 70
        self.rect.y = 480
        self.velocidade = 2

    def rotacionar_canhao(self, angulo):
        image = scale(load('images/canhao.png'), (15,30))
        self.angulo_da_mira += angulo
        self.angulo += angulo
        self.image = rotate(image, self.angulo_da_mira)
        self.rect = self.image.get_rect(center = self.rect.center)

    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP]:
            angulo = 1
            self.rotacionar_canhao(angulo)
                
        if keys[pygame.K_DOWN]:
            angulo = -1
            self.rotacionar_canhao(angulo)
            

        if keys[pygame.K_LEFT]:
            if self.rect.x > 0:
                self.rect.x -= self.velocidade

        if keys[pygame.K_RIGHT]:
            if self.rect.x < 187:
                self.rect.x += self.velocidade

class Bala(Sprite):
    def __init__(self, angulo=90, x=70, y = 480, center = None, velocidade=7):
        super().__init__()

        self.image = scale(load('images/bala.png'), (5,5))

        if center != None:
            self.rect = self.image.get_rect(center = center)
        else:
            self.rect = self.image.get_rect()

        self.angulo = angulo
        self.rect.x = x
        self.rect.y = y -10
        self.velocidade = velocidade
        self.velocidade_y = 5
        self.disparou = False
        self.gravidade = 0.1
        self.morreu = False


    def update(self):
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT] and self.disparou == False:
            if self.rect.x > 0:
                self.rect.x -= 2

        if keys[pygame.K_RIGHT] and self.disparou == False:
            if self.rect.x < 187:
                self.rect.x += 2

        if keys[pygame.K_SPACE]:
            self.disparou = True

        if self.disparou == False:
            if keys[pygame.K_UP]:
                self.angulo += 1

            if keys[pygame.K_DOWN]:
                self.angulo -= 1


        if self.disparou == True:
            if self.rect.x < 0 or self.rect.x > tamanho[0] or self.rect.y < 0 or self.rect.y > tamanho[1]:
                self.morreu = True

            v_x = ceil(cos(self.angulo * (pi / 180)) * self.velocidade)
            v_y = sin(self.angulo * (pi / 180)) * self.velocidade_y
            
            if sin(self.angulo * (pi / 180)) == 0 :
                if self.velocidade_y > 0 :
                    self.velocidade_y = 0

                v_y = self.velocidade_y

            self.rect.x = self.rect.x + v_x
            self.rect.y -= v_y
            self.velocidade_y -= self.gravidade
        
class Morro(Sprite):
    def __init__(self):
        super().__init__()

        self.image = scale(load('images/morro.png'), (100,400))
        self.rect = self.image.get_rect()
        self.rect.x = 300
        self.rect.y = 400

        
x_navio = randint(400 , 700)
state = x_navio
navio = Navio()
grupo_navio = GroupSingle(navio)
navio_inimigo = Navio_Inimigo(x_navio)
grupo_navio_inimigo = Group(navio_inimigo)
canhao = Canhao()
grupo_canhao = GroupSingle(canhao)
bala = Bala(center = canhao.rect.center)
grupo_bala = Group(bala)
grupo_mar = GroupSingle(Mar(superficie))


clock = Clock()
mortes = 0
rodada = 0
perdeu = False
pontos = 0

agent = DQN_Solver()
score = 0

nova_acao = True
action = None
errou = False
tolerancia = 800
n_acoes = 0
reset = False
done = False
episodio_quantidade = 1

agent.q_network.load_state_dict(torch.load('./net_2.pt'))
agent.exploration_rate = 0.01

while rodada < 100000:

    if reset == True:
        x_navio = randint(400 , 700)
        navio_inimigo.rect.x = x_navio
        state = x_navio
        reset = False

    if rodada % 1000 == 0:
        print(f'jogada: {rodada}   |   pontos: {pontos}   |  state: {state}')

    tentativa = 0
    while tentativa < episodio_quantidade:
    # Loop de eventos
        if rodada >=  1:
            clock.tick(60)  # FPS
        
        if nova_acao == True:
            acao_anterior = action

            velocidade, angulo, action = agent.choose_action(state)
            evitar_mesma_acao_contador = 0

            while action == acao_anterior:
                velocidade, angulo, action = agent.choose_action(state)

                if evitar_mesma_acao_contador == 10:
                    break

                evitar_mesma_acao_contador +=1

            rodada += 1

        nova_acao = False


        if canhao.angulo == angulo:
            bala.disparou = True

        if angulo > canhao.angulo:
            bala.angulo += 1
            canhao.rotacionar_canhao(1)
        
        if angulo < canhao.angulo:
            bala.angulo -= 1
            canhao.rotacionar_canhao(-1)

        for evento in event.get():  # Events
            if evento.type == QUIT:
                torch.save(agent.q_network.state_dict(), './net_3.pt')
                
                pygame.quit()

        if groupcollide(grupo_bala, grupo_navio, True, False):
            nova_acao = True
            agent.memory.add(state, action,0, state, True)
            agent.learn()

            angulo = canhao.angulo
            x = canhao.rect.x
            y = canhao.rect.y
            bala.kill()
            bala = Bala(angulo, x, y)
            grupo_bala.add(bala)
            reset = True
            tentativa += 1
            break

        if groupcollide(grupo_bala, grupo_mar, True, False):
            reward = navio_inimigo.rect.x - bala.rect.x

            if reward < 0:
                reward = -1
            else:
                reward = (1000 - reward) / 2000
                reward /= (tentativa+1)                
                        
            nova_acao = True

            if tentativa < (episodio_quantidade-1):
                done = False
            else:
                done = True
                reset = True

            agent.memory.add(state, action, -1, state, done)
            agent.learn()

            angulo = canhao.angulo
            x = canhao.rect.x
            y = canhao.rect.y
            bala.kill()
            bala = Bala(angulo, x, y)
            grupo_bala.add(bala)

            if errou == False:
                errou = True
            else: 
                errou = False

            tentativa += 1

        if bala.morreu == True:
            nova_acao = True

            if tentativa < (episodio_quantidade-1):
                done = False
            else:
                done = True
                reset = True

            agent.memory.add(state, action, -1, state, True)
            agent.learn()

            angulo = canhao.angulo
            x = canhao.rect.x
            y = canhao.rect.y
            bala.kill()
            bala = Bala(angulo, x, y)
            grupo_bala.add(bala)

            if errou == False:
                errou = True
            else: 
                errou = False

            tentativa += 1

        if groupcollide(grupo_bala, grupo_navio_inimigo, True, True):
            nova_acao = True

            pontos += 1
            angulo = canhao.angulo
            x = canhao.rect.x
            y = canhao.rect.y
            bala.kill()
            bala = Bala(angulo, x, y)
            grupo_bala.add(bala)

            if not perdeu:

                reward = 1 / (tentativa+1)

                x_navio = randint(400, 700) 
                navio_inimigo = Navio_Inimigo(x_navio)
                grupo_navio_inimigo.add(navio_inimigo)
                agent.memory.add(state, action, reward, x_navio, False)
                agent.learn()
            errou = False
            done = True

            Navio_Inimigo(x_navio)
            if tentativa >= (episodio_quantidade-1):
                reset = True

            tentativa += 1

        if rodada >= 1:
            superficie.blit(fundo, (0, 0))
            grupo_navio.draw(superficie)
            grupo_navio_inimigo.draw(superficie)
            grupo_canhao.draw(superficie)           
            grupo_mar.draw(superficie)

            if bala.disparou == True:
                grupo_bala.draw(superficie)
            

        grupo_navio.update()
        grupo_navio_inimigo.update()
        grupo_canhao.update()
        grupo_bala.update()
        grupo_mar.update()

        display.update()

        if done == True:
            break

torch.save(agent.q_network.state_dict(), './net_3.pt')
