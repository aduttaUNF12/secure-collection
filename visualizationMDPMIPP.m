foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results\';

%% average time plot -- security POW/POS

% for z = 1:3
% 
% if z == 1
%     X = categorical({'Diff 2', 'Diff 4', 'Insecure'});
%     X = reordercats(X,{'Diff 2', 'Diff 4', 'Insecure'});
% elseif z == 2
%     X = categorical({'Diff 2: TC = 2','Diff 2: TC = 3','Diff 2: TC = 4','Diff 2: TC = 5',...
%         'Diff 4: TC = 2','Diff 4: TC = 3','Diff 4: TC = 4','Diff 4: TC = 5','Insecure: TC = 2',...
%         'Insecure: TC = 3','Insecure: TC = 4','Insecure: TC = 5'});
%     X = reordercats(X,{'Diff 2: TC = 2','Diff 2: TC = 3','Diff 2: TC = 4','Diff 2: TC = 5',...
%         'Diff 4: TC = 2','Diff 4: TC = 3','Diff 4: TC = 4','Diff 4: TC = 5','Insecure: TC = 2',...
%         'Insecure: TC = 3','Insecure: TC = 4','Insecure: TC = 5'});
% else
%     X = categorical({'Diff 2: Range = 4','Diff 2: Range = 8','Diff 2: Range = 12','Diff 4: Range = 4',...
%         'Diff 4: Range = 8','Diff 4: Range = 12','Insecure: Range = 4','Insecure: Range = 8',...
%         'Insecure: Range = 12'});
%     X = reordercats(X,{'Diff 2: Range = 4','Diff 2: Range = 8','Diff 2: Range = 12','Diff 4: Range = 4',...
%         'Diff 4: Range = 8','Diff 4: Range = 12','Insecure: Range = 4','Insecure: Range = 8',...
%         'Insecure: Range = 12'});
% end
% allruns = [];
% figure();
% 
% % for n = 2:2:10
% n = 10;
%     if y > 1 & n < 10
%         continue;
%     end
%     
%     oneN = [];
%     for al = 2:2:6
%         for x = 1:12
%             
%         if z == 1
%             if x > 1
%                 continue;
%             end
%         elseif z == 2
%             if x < 2 || x > 5
%                 continue;
%             end
%         elseif z == 3
%             if x ~= 4 & x ~= 8 & x ~= 12
%                 continue;
%             end
%         end    
%         
%         for y = 1:3
% %         if y == 1
% %             foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (CC)\';
% %         elseif y == 2
% %             foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersCC)\';
% %         else
% %             foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersCC)\';
% %         end
% 
%         if y == 1
%             if z == 1
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (CC)\';
%             elseif z == 2
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (PC)\';
%             else
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (OC)\';
%             end
%         elseif y == 2
%             if z == 1
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersCC)\';
%             elseif z == 2
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersPC)\';
%             else
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersOC)\';
%             end
%         else
%             if z == 1
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersCC)\';
%             elseif z == 2
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersPC)\';
%             else
%                 foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersOC)\';
%             end
%         end
%         
%         if al==1
%             titleS = 'PoW-Diff_1';
%             
%             if z == 1
%             strm = strcat(foldersecure,'PoW-Diff_1-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-Diff_1-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-Diff_1-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             end
%         end
%         
%         if al==2
%             titleS = 'PoW-Diff_2';
%             
%             if z == 1
%                 strm = strcat(foldersecure,'PoW-Diff_2-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-Diff_2-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-Diff_2-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             end
%         end
%         
%         if al==3
%             titleS = 'PoW-Diff_3';
%             if z == 1
%                 strm = strcat(foldersecure,'PoW-Diff_3-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-Diff_3-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-Diff_3-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             end
%         end
%         
%         if al==4
%             titleS = 'PoW-Diff_4';
%             if z == 1
%                 strm = strcat(foldersecure,'PoW-Diff_4-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-Diff_4-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-Diff_4-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             end
%         end
%         
%         if al==5
%             titleS = 'PoW-Diff_5';
%             
%             if z == 1
%                 strm = strcat(foldersecure,'PoW-Diff_5-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-Diff_5-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-Diff_5-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
%             end
%         end
%         
%         if al==6
%             titleS = 'Insecure';
%             
%             if z == 1
%                 strm = strcat(foldersecure,'PoW-_CC_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
%             elseif z == 2
%                 strm = strcat(foldersecure,'PoW-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
%             else
%                 strm = strcat(foldersecure,'PoW-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
%             end
%         end
%         
%         %         if al==2
%         %             titleS = 'PoS';
%         %             strm = strcat(foldersecure,'PoS-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         %         end
%         
%         yy1 = [];
%         
%         %file name --> PoW-Greedy-CC2_Secure-WithAttacks(small)__run_5
%         % insecure --> PoW-Greedy-CC2_NotSecure-WithAttacks(small)__run_3
%         
%         for run =1:10
%             if y == 1
%                 str = strcat(strm,'_run_',num2str(run),'.mat');
%             elseif y == 2
%                 str = strcat(strm,'2Attackers_run_',num2str(run),'.mat');
%             else
%                 str = strcat(strm,'4Attackers_run_',num2str(run),'.mat');
%             end
%             load(str);
%             yy1 = [yy1; timeSec];
%         end
%         
%         yy = mean(yy1);
%         oneN = [oneN yy];
%         plot(mean(allruns));
%         hold all;
%         clearvars allMSE allReward allVar timeSec allPaths allCC Pred_n;
%         end
%         
%         
%     allruns = [allruns; oneN];
%     oneN = [];
%     hold off;
%         end
% 
%     end
%     
%     
% % end
% 
% bar(X, allruns);
% set(gca,'YScale','lin')
% ylabel('Time (sec.)','FontSize',14);
% % xlabel('Number of robots','FontSize',14);
% % legend('PoW: Diff 1','PoW: Diff 2','PoW: Diff 3','PoW: Diff 4','PoW: Diff 5','Insecure','Location','best');
% if z == 1
%     xlabel('Blockchain Type (CC, 10 Robots)','FontSize',14);
% %     legend('PoW: Diff 2','PoW: Diff 4','Insecure');
%     legend('1 Attacker','2 Attackers','4 Attackers');
% 
% elseif z == 2
%     xlabel('Blockchain Type (PC, 10 Robots)','FontSize',14);
% %     legend('Diff 2, TC = 2','Diff 2, TC = 3','Diff 2, TC = 4','Diff 2, TC = 5','Diff 4, TC = 2',...
% %         'Diff 4, TC = 3','Diff 4, TC = 4','Diff 4, TC = 5','Insecure, TC = 2','Insecure, TC = 3',...
% %         'Insecure, TC = 4','Insecure, TC = 5');
%     legend('1 Attacker','2 Attackers','4 Attackers');
% 
% else
%     xlabel('Blockchain Type (OC, 10 Robots)','FontSize',14);
% %     legend('Diff 2, Range = 4','Diff 2, Range = 8','Diff 2, Range = 12','Diff 4, Range = 4',...
% %         'Diff 4, Range = 8','Diff 4, Range = 12','Insecure, Range = 4','Insecure, Range = 8',...
% %         'Insecure, Range = 12');
%     legend('1 Attacker','2 Attackers','4 Attackers');
% end
% 
% filename = strcat('timecomp.png');
% %saveas(gcf,filename);
% end

%% average MSE plot -- POW and POS against insecure and no-attack.

y=1;

for z = 1
for n = 8
    figure();
    if y > 1 & n < 10
        continue;
    end

%     a = 1;
%     b = 1;
%     c = 1;
%     d = 1;
%     e = 1;
%     f = 1;
%     g = 1;
    
    for al = 1:7
        
%         if al == 5
%             continue;
%         end
        if al > 7
            al = 7;
        end
            
        for x = 1
        
        if z == 1
            if x > 1
                continue;
            end
        elseif z == 2
            if x < 2 || x > 5
                continue;
            end
        elseif z == 3
            if x ~= 4 & x ~= 8 & x ~= 12
                continue;
            end
        end
        a = 1;
        b = 1;
        c = 1;
        d = 1;
        e = 1;
        f = 1;
        g = 1;
        for y = 1
            
        if y == 1
            if z == 1
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (CC)\';
            elseif z == 2
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (PC)\';
            else
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (OC)\';
            end
        elseif y == 2
            if z == 1
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersCC)\';
            elseif z == 2
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersPC)\';
            else
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (2AttackersOC)\';
            end
        else
            if z == 1
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersCC)\';
            elseif z == 2
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersPC)\';
            else
                foldersecure = 'C:\Users\Tamim\Documents\UNF\Graduate\Research\Greedy MIPP code base\Results (4AttackersOC)\';
            end
        end
        
        if al==1
            titleS = 'PoW-Diff_1';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-Diff_1-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-Diff_1-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-Diff_1-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            end
        end
        
        if al==2
            titleS = 'PoW-Diff_2';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-Diff_2-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-Diff_2-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-Diff_2-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            end
        end
        
        if al==3
            titleS = 'PoW-Diff_3';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-Diff_3-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-Diff_3-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-Diff_3-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            end
        end
        
        if al==4
            titleS = 'PoW-Diff_4';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-Diff_4-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            elseif z == 2 
                strm = strcat(foldersecure,'PoW-Diff_4-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-Diff_4-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            end
        end
        
        if al==5
            titleS = 'PoW-Diff_5';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-Diff_5-_CC_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-Diff_5-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-Diff_5-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_Secure-WithAttacks(small)_');
            end
        end
        
        if al==6
            titleS = 'Insecure';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-_CC_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
            else
                strm = strcat(foldersecure,'PoW-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)_');
            end
        end
        
        if al==7 && y == 1
            titleS = 'No attack';
            
            if z == 1
                strm = strcat(foldersecure,'PoW-_CC_Greedy-CC',num2str(n),'_NotSecure-NoAttacks_');
            elseif z == 2
                strm = strcat(foldersecure,'PoW-_PC_turn(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-NoAttacks_');
            else
                strm = strcat(foldersecure,'PoW-_OC_range(',num2str(x),')_Greedy-CC',num2str(n),'_NotSecure-NoAttacks_');
            end
        end
        
        %         if al==4 % not used PoS
        %             titleS = 'PoS';
        %             strm = strcat(foldersecure,'PoS-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
        %         end
        
        allruns = [];
        
        for run = 1:10
            if y == 1
                str = strcat(strm,'_run_',num2str(run),'.mat');
            elseif y == 2
                str = strcat(strm,'2Attackers_run_',num2str(run),'.mat');
            else
                str = strcat(strm,'4Attackers_run_',num2str(run),'.mat');
            end
            
            if contains(str, "NoAttacks_2Attackers") || contains(str, "NoAttacks_4Attackers")
                continue;
            end
            
            load(str);
            avgMSE = (sum(cell2mat(allMSE')))/n ;
            allruns = [allruns; avgMSE];
%             if al == 2
%                 fprintf("Difficulty %d, Robots = %d, Run %d, Mistakes = %d\n",al,n,run,mistakes);
%             end
        end
        
        if al==1
            stdv = std(allruns);
            xx = 1:1:numel(stdv);
            
            if a == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','bs-','patchSaturation',0.03);
            elseif a == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','b^-','patchSaturation',0.03);
            elseif a == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','bo-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','bx-','patchSaturation',0.03);
            end
            a = a + 1;
            
            %plot(mean(allruns),'bo-');
        end
        
        if al==2
            stdv = std(allruns);
            xx = 1:1:numel(stdv);
            
            if b == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ms-','patchSaturation',0.03);
            elseif b == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','m^-','patchSaturation',0.03);
            elseif b == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','mo-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','mx-','patchSaturation',0.03);
            end
            b = b + 1;
            
            %plot(mean(allruns),'mx-');
        end
        
        if al==3
            stdv = std(allruns);
            xx = 1:1:numel(stdv);
            
            if c == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ys-','patchSaturation',0.03);
            elseif c == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','y^-','patchSaturation',0.03);
            elseif c == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','yo-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','yx-','patchSaturation',0.03);
            end
            c = c + 1;
            
            %plot(mean(allruns),'yv-');
        end
        
        if al==4
            stdv = std(allruns);
            xx = 1:1:numel(stdv);
            
            if z == 1
            if d == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ks-','patchSaturation',0.03);
            elseif d == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','k^-','patchSaturation',0.03);
            elseif d == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ko-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','kx-','patchSaturation',0.03);
            end
            elseif z == 2
            if d == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ks-','patchSaturation',0.03);
            elseif d == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','k^-','patchSaturation',0.03);
            elseif d == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ko-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','kx-','patchSaturation',0.03);
            end
            else
            if d == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ks-','patchSaturation',0.03);
            elseif d == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','k^-','patchSaturation',0.03);
            elseif d == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ko-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','kx-','patchSaturation',0.03);
            end
            end
            d = d + 1;
            
            %plot(mean(allruns),'ks-');
        end
        
        if al==5
            stdv = std(allruns);
            xx = 1:1:numel(stdv);
            
            if e == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','cs-','patchSaturation',0.03);
            elseif e == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','c^-','patchSaturation',0.03);
            elseif e == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','co-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','cx-','patchSaturation',0.03);
            end
            e = e + 1;
            
            %plot(mean(allruns),'c*-');
            M = max(stdv, [], 'all');
%             fprintf("Standard Deviation = %f\n",M);
        end
        
        if al==6
            if f == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','rs-','patchSaturation',0.03);
            elseif f == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','r^-','patchSaturation',0.03);
            elseif f == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','ro-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','rx-','patchSaturation',0.03);
            end
            f = f + 1;
        end
        
        if al==7 && y == 1
            if g == 1
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','gs-','patchSaturation',0.03);
            elseif g == 2
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','g^-','patchSaturation',0.03);
            elseif g == 3
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','go-','patchSaturation',0.03);
            else
                shadedErrorBar(xx,mean(allruns),stdv,'lineprops','gx-','patchSaturation',0.03);
            end
            g = g + 1;
        end
        
        hold all;
        clearvars allMSE allReward allVar timeSec allPaths allCC Pred_n;
        end
        
        end
    end
    
    hold off;
    ylabel('$\mathcal{F}$','Interpreter','latex','FontSize',14);
    xlabel('Path length','FontSize',14);
    plotT = strcat('n=',num2str(n));
    title(plotT,'FontSize',14);
    legend('PoW: Diff 1','PoW: Diff 2','PoW: Diff 3','PoW: Diff 4','PoW: Diff 5','Insecure','No Attack');
%     legend('CC','PC: Turn Count = 2','PC: Turn Count = 3','PC: Turn Count = 4','PC: Turn Count = 5',...
%         'OC: Range 4','OC: Range 8','OC: Range 12');
%     legend('Diff 3: 1 Attacker','Diff 3: 2 Attackers','Diff 3: 4 Attackers','Insecure: 1 Attacker',...
%        'Insecure: 2 Attackers','Insecure: 4 Attackers','No Attack');
    filename = strcat('msecomp_w_noattack_n',num2str(n),'.png');
    %saveas(gcf,filename);
end
end
    



% %% average variance plot -- POW and POS against insecure and no-attack.
% 
% allruns=[];
% 
% for n =2:2:10
% 
%     figure();
% 
%     for al = 1:7
%         
%         if al==1
%             titleS = 'PoW-Diff_1';
%             strm = strcat(foldersecure,'PoW-Diff_1-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==2
%             titleS = 'PoW-Diff_2';
%             strm = strcat(foldersecure,'PoW-Diff_2-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==3
%             titleS = 'PoW-Diff_3';
%             strm = strcat(foldersecure,'PoW-Diff_3-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==4
%             titleS = 'PoW-Diff_4';
%             strm = strcat(foldersecure,'PoW-Diff_4-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==5
%             titleS = 'PoW-Diff_4';
%             strm = strcat(foldersecure,'PoW-Diff_4-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==6
%             titleS = 'Insecure';
%             strm = strcat(foldersecure,'PoW-Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)__');
%         end
% 
%         if al==7
%             titleS = 'No attack';
%             strm = strcat(foldersecure,'PoW-Greedy-CC',num2str(n),'_NotSecure-NoAttacks__');
%         end        
%         
%         %         if al==4 % not used PoS
%         %             titleS = 'PoS';
%         %             strm = strcat(foldersecure,'PoS-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         %         end
%         
%         allruns = [];
% 
%         for run =1:5
%             str = strcat(strm,'run_',num2str(run),'.mat');
%             load(str);
%             out = mean(mean(cat(3,allVar{:}),3));
%             allruns = [allruns; out];
%         end
%         
%         if al==1
%             %shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'bo-');
%         end
% 
%         if al==2
%             %shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'mx-');
%         end
%         
%         if al==3
%             %shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'yv-');
%         end
%         
%         if al==4
%             %shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'ks-');
%         end
%         
%         if al==5
%             %shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'c*-');
%         end
%         
%         if al==6
%             plot(mean(allruns),'rd-');
%         end
%         
%         if al==7
%             plot(mean(allruns),'g+-');
%         end
% 
%         hold all;
%         clearvars allMSE allReward allVar timeSec allPaths allCC Pred_n;
%     end
% 
%     hold off;
%     ylabel('Average Variance','FontSize',14);
%     xlabel('Path length','FontSize',14);
%     plotT = strcat('n=',num2str(n));
%     title(plotT,'FontSize',14);
%     legend('PoW: Diff 1','PoW: Diff 2','PoW: Diff 3','PoW: Diff 4', 'PoW: Diff 5', 'Insecure','No Attack','FontSize',10,'Location','best')
%     filename = strcat('varcomp_w_noattack_n',num2str(n),'.png');
%     %saveas(gcf,filename);
% end




% % average MSE plot -- POW and POS against insecure and no-attack.
% 
% for n =2:2:10
%     
%     figure();
%     
%     for al = 1:7
%         
%         if al==1
%             titleS = 'PoW-Diff_1';
%             strm = strcat(foldersecure,'PoW-Diff_1-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==2
%             titleS = 'PoW-Diff_2';
%             strm = strcat(foldersecure,'PoW-Diff_2-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==3
%             titleS = 'PoW-Diff_3';
%             strm = strcat(foldersecure,'PoW-Diff_3-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==4
%             titleS = 'PoW-Diff_4';
%             strm = strcat(foldersecure,'PoW-Diff_4-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==5
%             titleS = 'PoW-Diff_5';
%             strm = strcat(foldersecure,'PoW-Diff_5-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         end
%         
%         if al==6
%             titleS = 'Insecure';
%             strm = strcat(foldersecure,'PoW-Greedy-CC',num2str(n),'_NotSecure-WithAttacks(small)__');
%         end
%         
%         if al==7
%             titleS = 'No attack';
%             strm = strcat(foldersecure,'PoW-Greedy-CC',num2str(n),'_NotSecure-NoAttacks__');
%         end
%         
%         %         if al==4 % not used PoS
%         %             titleS = 'PoS';
%         %             strm = strcat(foldersecure,'PoS-Greedy-CC',num2str(n),'_Secure-WithAttacks(small)__');
%         %         end
%         
%         allruns = [];
%         
%         for run =1:5
%             str = strcat(strm,'run_',num2str(run),'.mat');
%             load(str);
%             avgMSE = (sum(cell2mat(allMSE')))/n ;
%             allruns = [allruns; avgMSE];
%         end
%                 
%         if al==5
%             stdv = std(allruns);
%             xx = 1:1:numel(stdv);
%             shadedErrorBar(xx,mean(allruns),stdv,'lineprops','-b','patchSaturation',0.03);
%             plot(mean(allruns),'c*-');
%         end
%         
%         if al==6
%             plot(mean(allruns),'rd-');
%         end
%         
%         if al==7
%             plot(mean(allruns),'g+-');
%         end
%         
%         
%         hold all;
%         if al>=5
%             clearvars allMSE allReward allVar timeSec allPaths allCC Pred_n;
%         end
%     end
%     
%     hold off;
%     ylabel('Average MSE','FontSize',14);
%     xlabel('Path length','FontSize',14);
%     plotT = strcat('n=',num2str(n));
%     title(plotT,'FontSize',14);
%     legend('a','PoW', 'Insecure','No Attack','FontSize',10,'Location','best')
%     filename = strcat('msecomp_w_noattack_n',num2str(n),'.png');
%     %saveas(gcf,filename);
% end