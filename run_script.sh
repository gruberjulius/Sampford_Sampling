# pip uninstall fairseq
# pip install --editable ./

#Stochastic Beam Search
# for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
# do
#     echo "sbs"
#     echo  $i.txt
#     CUDA_VISIBLE_DEVICES=0 fairseq-generate --stochastic-beam-search data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --beam 5 --nbest 5 --sampling-temperature $i --no-early-stop --unnormalized --batch-size 16 > outputs/sbs/5/$i.txt
# done

# Sampling w/ topk option
# for j in 5 10 20
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#     do
#         echo "sampling"
#         echo  $j
#         echo  $i.txt
#         CUDA_VISIBLE_DEVICES=0 fairseq-generate --sampling data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --beam $j --nbest $j --sampling-temperature $i --no-early-stop --unnormalized --batch-size 16 --sampling-topk $j > outputs/sampling/$j/$i.txt
#     done
# done

# Diverse Beam Search
# for j in 5 10 20
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#     do
#         echo "diverse"
#         echo  $j
#         echo  $i.txt
#         CUDA_VISIBLE_DEVICES=0 fairseq-generate --diverse-beam-groups $j data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --beam $j --nbest $j --diverse-beam-strength $i --no-early-stop --unnormalized --batch-size 16 > outputs/diverse/$j/$i.txt
#     done
# done

#Beam Search
# for j in 5 10 20
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#     do
#         echo "beam"
#         echo  $j
#         echo  $i.txt
#         CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --beam $j --nbest $j --no-early-stop --unnormalized --sampling-temperature $i --batch-size 16 > outputs/bs/$j/$i.txt
#     done
# done

#Conditional Poission Beam Search
# for j in 5 10 20
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#     do
#         echo "beam"
#         echo  $j
#         echo  $i.txt
#         CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --cps --beam $j --nbest $j --unnormalized --batch-size 16  --sampling-temperature $i > outputs/cps/$j/$i.txt
#     done
# done


#Sampford
# for j in 5 10 20
# do
#     for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8
#     do
#         echo "beam"
#         echo  $j
#         echo  $i.txt
#         CUDA_VISIBLE_DEVICES=0 fairseq-generate data-bin/wmt14.v2.en-fr.newstest2014/wmt14.en-fr.newstest2014 --path models/wmt14.v2.en-fr.fconv-py/wmt14.en-fr.fconv-py/model.pt --sampford --beam $j --nbest $j --unnormalized --batch-size 16  --sampling-temperature $i > outputs/sampford/$j/$i.txt
#     done
# done

