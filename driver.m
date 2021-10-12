%% common initial data fo both the algorithms
clc
clear
M=14;% for a MxM environment.
X_train = (1:M*M)';
init_trn_data = 0.1;% percentage of all nodes in the environment.
secure = 1; %0 = insecure, 1 = secure (blockchain)
attack = 3; %0 = no attack, 1 = random attack, 2 = zero attack, 3 = small attack
difficulty = 1;
type = 1; %0 = CC, 1 = PC, 2 = OC
turnPC = 2;
rangeOC = 4;

tic
for n=10
    for run = 1:10
        x_active = randsample(X_train,ceil(init_trn_data*M*M));%select k% random points to initialize GP hyperparameters.
        range = 0.30;
        
        %% pre-processing -- K-means clustering for partitioining the region -- common variable.
        %id = 1;
        [idx,centr] = find_partitions(n,M,0);% params: (robot#,size,display_flag)
        
        for turnPC = 2:5
            greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack, difficulty, type, turnPC, rangeOC);
            pause(1);
            clc
        end
        
%         for secure = 0:1
%             for attack = 0:3:3
%                 if secure == 1
%                     for difficulty = 1:5
%                         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack, difficulty, type, turnPC, rangeOC);
%                         pause(1);
%                         clc
%                     end                
%                 else
%                     greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack, difficulty, type, turnPC, rangeOC);
%                     pause(1);
%                     clc
%                 end
%             end
%         end
        

%         %% insecure, no attack
%         secure = 0;
%         attack = 0;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%         
%         %% insecure, random attack
%         secure = 0;
%         attack = 1;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%         
%         
%         %% insecure, zero attack
%         secure = 0;
%         attack = 2;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%         
%         
%         %% secure, no attack
%         secure = 1;
%         attack = 0;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%         
%         
%         %% secure, random attack
%         secure = 1;
%         attack = 1;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%         
%         
%         %% secure, zero attack
%         secure = 1;
%         attack = 2;
%         greedy(n,M,x_active,0,1,0,idx, centr, range,run, secure, attack);
%         pause(1);
%         clc
%           
%         
    end
end
disp(toc);
clear;