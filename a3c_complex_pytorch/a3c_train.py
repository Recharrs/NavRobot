import torch.optim as optim

from models import *
from torch.autograd import Variable

import logging

from my_utils import *

import math

def ensure_shared_grads(rank, model, shared_model):
    global_norm = 0
    for param, shared_param in zip(model.parameters(),shared_model.parameters()):
        if shared_param.grad is not None:
            return
        #if rank == 0:
        #    print(param.size(), torch.norm(param.grad, 2).cpu().data.numpy())
        #global_norm += torch.norm(param.grad, 2) ** 2
        shared_param._grad = param.grad
    #global_norm = torch.sqrt(global_norm)
    #print(global_norm)

def train(rank, args, shared_model):
    torch.manual_seed(args.seed + rank)

    env = env_wrapper(args.app_location, idx=rank, base=256)
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    model = A3C_LSTM_GA(args)
    model = model.cuda()
    
    # movie maker
    my_movie_maker = movie_maker(path=args.vedio_location)
    #####

    if (args.load != "0"):
        print(str(rank) + " Loading model ... "+args.load)
        model.load_state_dict(
            torch.load(args.load, map_location=lambda storage, loc: storage))

    model.train()
    optimizer = optim.SGD(shared_model.parameters(), lr=args.lr)

    p_losses = []
    v_losses = []

    (image, instruction), _, _, _ = env.reset()
    instruction_idx = []
    for word in instruction.split(" "):
        instruction_idx.append(env.word_to_idx[word])
    instruction_idx = np.array(instruction_idx)

    image = torch.from_numpy(image).float().cuda()
    instruction_idx = torch.from_numpy(instruction_idx).view(1, -1).cuda()

    done = True

    episode_length = 0
    num_iters = 0
    
    # export
    episode = 0
    times = 0
    instruction_prev = instruction
    ##### export done

    while True:
        import time
        time.sleep(0.3)
        
        # Sync with the shared model
        model.load_state_dict(shared_model.state_dict())
        if done:
            episode_length = 0
            cx = Variable(torch.zeros(1, 256)).cuda()
            hx = Variable(torch.zeros(1, 256)).cuda()

        else:
            cx = Variable(cx.data).cuda()
            hx = Variable(hx.data).cuda()

        values = []
        log_probs = []
        rewards = []
        entropies = []
        
        
        
        for step in range(args.num_steps):
            episode_length += 1
            tx = Variable(torch.from_numpy(np.array([episode_length])).long()).cuda()
            
            value, logit, (hx, cx) = model((Variable(image.unsqueeze(0)).cuda(),
                                            Variable(instruction_idx).cuda(),
                                            (tx, hx, cx)))

            prob = F.softmax(logit, dim=1)
            log_prob = F.log_softmax(logit, dim=1)
            entropy = -(log_prob * prob).sum(1)
            entropies.append(entropy)

            action = prob.multinomial().data
            log_prob = log_prob.gather(1, Variable(action))

            action = action.cpu().numpy()[0, 0]
            (image, instruction), reward, done,  _ = env.step(action)
            
            instruction_idx = []
            for word in instruction.split(" "):
                instruction_idx.append(env.word_to_idx[word])
            instruction_idx = np.array(instruction_idx)
            instruction_idx = torch.from_numpy(
                    instruction_idx).view(1, -1)
            
            reward = -0.2 if reward == 0.2 else reward
            done = done or episode_length >= args.max_episode_length

            # export
            if rank == 0:
                if reward == 1.0:
                    cx = Variable(torch.zeros(1, 256)).cuda()
                    hx = Variable(torch.zeros(1, 256)).cuda()
                    times += 1
                if done:
                    episode += 1
                    my_movie_maker.export_ani(instruction_prev, reward, times)
                    if episode % 1000 == 0:
                        torch.save(model.state_dict(), args.model_location)
                    times = 0
                else:
                    my_movie_maker.add_new_image(image, instruction)
            ##### export done

            if done:
                (image, instruction), _, _, _ = env.reset()                
                instruction_idx = []
                for word in instruction.split(" "):
                    instruction_idx.append(env.word_to_idx[word])
                instruction_idx = np.array(instruction_idx)
                instruction_idx = torch.from_numpy(
                        instruction_idx).view(1, -1)

                # bug issue
                instruction_prev = instruction
                ##### issue done

            image = torch.from_numpy(image).float().cuda()

            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if done:
                break

        R = torch.zeros(1, 1)
        if not done:
            tx = Variable(torch.from_numpy(np.array([episode_length])).long()).cuda()
            value, _, _ = model((Variable(image.unsqueeze(0)).cuda(),
                                 Variable(instruction_idx).cuda(), 
                                 (tx, hx, cx)))
            R = value.data.cuda()

        values.append(Variable(R))
        policy_loss = 0
        value_loss = 0
        R = Variable(R)

        gae = torch.zeros(1, 1)
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - 0.01 * entropies[i]

        optimizer.zero_grad()

        p_losses.append(policy_loss.data[0, 0])
        v_losses.append(value_loss.data[0, 0])

        if(len(p_losses) > 1000):
            num_iters += 1
            print(" ".join([
                  "Training thread: {}".format(rank),
                  "Num iters: {:.1f}K".format(num_iters),
                  "Avg policy loss: {:.4f}".format(np.mean(p_losses)),
                  "Avg value loss: {:.4f}".format(np.mean(v_losses))
                  ]))
            logging.info(" ".join([
                  "Training thread: {}".format(rank),
                  "Num iters: {:.1f}K".format(num_iters),
                  "Avg policy loss: {:.4f}".format(np.mean(p_losses)),
                  "Avg value loss: {:.4f}".format(np.mean(v_losses))
                  ]))
            p_losses = []
            v_losses = []

        (policy_loss + 0.5 * value_loss).backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 40)

        ensure_shared_grads(rank, model, shared_model)
        optimizer.step()
