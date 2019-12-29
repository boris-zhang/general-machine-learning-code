# 建立中间表
CREATE TABLE `ai_keywords_mid_classification_test` (
  `seqno` bigint(20) NOT NULL,
  `loc` int(11) NOT NULL,
  `class_level1` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '第3名一级品类',
  `class_level2` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '第3名二级品类',
  `class_level3` varchar(50) COLLATE utf8mb4_unicode_ci DEFAULT NULL COMMENT '第3名三级品类',
  PRIMARY KEY (`seqno`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


# 选出关联结果及所处位置，因为要选取最后一个词出现的位置，因此用reverse函数
select locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
+ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0) as loc
,t1.seqno,t2.class_level1,t2.class_level2,t2.class_level3
     from ai_keywords_classification_test t1
      inner join ai_keywords_classification_classdef t2 
       on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
          or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
      where t2.class_level3 is not null and t1.seqno in (8,9,23,24);

# mysql获取分组后每组的最小值
select a.* from test a
inner join (select seqno,max(loc) as loc from test group by seqno) b
    on a.seqno=b.seqno and a.loc=b.loc
order by a.seqno;


# 选出最小位置对应的结果，插入中间表
insert into ai_keywords_mid_classification_test(seqno,loc,class_level1,class_level2,class_level3)
select a.seqno,a.loc,a.class_level1,a.class_level2,a.class_level3
from
(select locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
+ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0) as loc
,t1.seqno,t2.class_level1,t2.class_level2,t2.class_level3
     from ai_keywords_classification_test t1
      inner join ai_keywords_classification_classdef t2 
       on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
          or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
      where t2.class_level3 is not null and t1.predict_method ='bayes') a 
inner join 
(select tt.seqno,min(tt.loc) as loc from
(select locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
+ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0) as loc
,t1.seqno,t2.class_level1,t2.class_level2,t2.class_level3
     from ai_keywords_classification_test t1
      inner join ai_keywords_classification_classdef t2 
       on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
          or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
      where t2.class_level3 is not null and t1.predict_method ='bayes') tt group by tt.seqno) b 
on a.seqno=b.seqno and a.loc=b.loc
group by a.seqno,a.loc
order by a.seqno;

# 更新
update ai_keywords_classification_test t1
inner join ai_keywords_mid_classification_test t2
  on t1.seqno=t2.seqno
set t1.class_level1=t2.class_level1
  ,t1.class_level2=t2.class_level2
  ,t1.class_level3=t2.class_level3
  ,t1.predict_method='sql_segwords_multi'
where t1.predict_method='bayes';


    ## 同一序号广告语中包含分类关键词的数量
    select t1.seqno,count(distinct locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
    +ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0))
    from ai_keywords_classification_test t1
    inner join ai_keywords_classification_classdef t2 
        on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
            or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
    where t2.class_level3 is not null and t1.predict_method='sql_segwords_multi'
    group by t1.seqno;
    
      
    update ai_keywords_mid_classification_test A
    left join (
        select t1.seqno,count(distinct locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
        +ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0)) as num
        from ai_keywords_classification_test t1
        inner join ai_keywords_classification_classdef t2 
            on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
                or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
            and t2.search_word_flag=1
        where t2.class_level3 is not null and t1.predict_method='sql_segwords_multi'
        group by t1.seqno) B
    on A.seqno=B.seqno
    set A.num=B.num;
    


#------ 存储过程 ------#
CREATE DEFINER=`kdmsu`@`%` PROCEDURE `ai_keywords_classification`()
BEGIN
## 清空中间表
truncate table ai_keywords_mid_classification_test;
## 选出最小位置对应的结果，插入中间表
insert into ai_keywords_mid_classification_test(seqno,loc,class_level1,class_level2,class_level3)
select a.seqno,a.loc,a.class_level1,a.class_level2,a.class_level3
from
(select locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
+ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0) as loc
,t1.seqno,t2.class_level1,t2.class_level2,t2.class_level3
     from ai_keywords_classification_test t1
      inner join ai_keywords_classification_classdef t2 
       on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
          or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
      where t2.class_level3 is not null and t1.predict_method ='bayes') a 
inner join 
(select tt.seqno,min(tt.loc) as loc from
(select locate(reverse(t2.class_level31_proc),reverse(t1.keyword_segmented_cutall_proc))
+ifnull(locate(reverse(t2.class_level32_proc),reverse(t1.keyword_segmented_cutall_proc)),0) as loc
,t1.seqno,t2.class_level1,t2.class_level2,t2.class_level3
     from ai_keywords_classification_test t1
      inner join ai_keywords_classification_classdef t2 
       on (locate(t2.class_level31_proc,t1.keyword_segmented_cutall_proc)>0 
          or locate(t2.class_level32_proc,t1.keyword_segmented_cutall_proc)>0) 
        and t2.search_word_flag=1
      where t2.class_level3 is not null and t1.predict_method ='bayes') tt group by tt.seqno) b 
on a.seqno=b.seqno and a.loc=b.loc
group by a.seqno,a.loc
order by a.seqno;
# 更新
update ai_keywords_classification_test t1
inner join ai_keywords_mid_classification_test t2
  on t1.seqno=t2.seqno
set t1.class_level1=t2.class_level1
  ,t1.class_level2=t2.class_level2
  ,t1.class_level3=t2.class_level3
  ,t1.predict_method='sql_segwords_multi'
where t1.predict_method='bayes';
END



