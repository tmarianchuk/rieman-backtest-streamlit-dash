WITH map AS (SELECT distinct
  o.name AS org,
  c.name as chan
  FROM eligo_portal_organizations o
  JOIN eligo_portal_sales_channels c ON o.sales_channel_id = c.id
  JOIN eligo_portal_categories ca ON o.category_id = ca.id)

  , cbase AS (
    SELECT
      map.chan :: VARCHAR AS org_type,
      cb.request_id,
      aa.aggregation_id,
      cb.service_zone,
      cb.account_number,
      cb.term,
      cb.enroll_on :: DATE   AS enroll_on, case when cb.sales_price <= 0 then null else cb.sales_price end as contract_rate
    FROM commercial_base cb
      LEFT JOIN aggregation_accounts aa ON aa.pricing_request_id = cb.request_id
      JOIN map ON map.org = cb.sales_organization
    WHERE
      cb.request_id IS NOT NULL AND cb.service_zone IS NOT NULL AND cb.account_number IS NOT NULL AND cb.enroll_on IS NOT NULL AND
      cb.term IS NOT NULL AND cb.sales_organization != 'Legacy CRM')
  ,
    agga AS (select
'Aggregation' :: VARCHAR AS org_type,
coalesce(pricing_request_id,-aggregation_account_id)    AS request_id,
ac.aggregation_id,
service_zone,
utility_account_number as account_number,
coalesce(term,18)::INT as term,
coalesce(request_completion_date,enroll_on,start_date, '2015-06-01'::DATE)::DATE as enroll_on
  from aggregation_customers ac
  left join aggregation_accounts aa on aa.id = ac.aggregation_account_id
  where (ac.type_of_mail_to_send = 'opt_out' and ac.opted_out = False) or (ac.type_of_mail_to_send = 'opt_in' and ac.opted_in = True))
  ,

    olds AS (SELECT
               CASE WHEN campaigns.name ~* 'callfirm|cep2|cep3|tcf'
                 THEN 'PAM'
               WHEN campaigns.name ~*
                    'cdg|cea|pwrcompny|savewave|utilitysa|electron|dirconpro|cepartners|uspowertrade|g5tm'
                 THEN 'Broker'
               ELSE 'Inside Sales' END  AS org_type,
               (-leads.id)   AS request_id,
               sz.name                  AS service_zone,
               cf_account_num           AS account_number,
               leads.created_at :: DATE AS enroll_on,
               rates.term AS term, case when rates.rate_kwh <= 0 then null else rates.rate_kwh end as contract_rate
             FROM leads
               JOIN service_zones sz ON sz.id = leads.service_zone_id
               JOIN campaigns ON leads.campaign_id = campaigns.id
               join rates on rates.name = leads.cf_rate_class
             WHERE cf_account_num IS NOT NULL AND leads.created_at IS NOT NULL),

    base AS (SELECT DISTINCT ON (service_zone, account_number, request_id)
               org_type,
               request_id,
               aggregation_id,
               service_zone,
               account_number,
               term,
               enroll_on

             FROM (SELECT
                     org_type,
                     request_id,
                     aggregation_id,
                     service_zone,
                     account_number,
                     term,
                     enroll_on,
                     1 AS level
                   FROM cbase

               UNION SELECT
                           org_type,
                           request_id,
                           aggregation_id,
                           service_zone,
                           account_number,
                           term,
                           enroll_on,
                           2 AS level
                         FROM agga

               UNION SELECT
                           org_type,
                           request_id,
                           NULL AS aggregation_id,
                           service_zone,
                           account_number,
                           term,
                           enroll_on,
                           3 AS level
                         FROM olds) foo
             ORDER BY service_zone, account_number, request_id, level ASC),



    unbills AS (SELECT
                  ur.service_zone,
                  split_part(ur.service_zone, '_', 2)          AS state_name,
                  ur.utility_account_number,
                  date_trunc('month', allocated_month) :: DATE AS month,
                  adjusted_unbilled_usage                 AS billable_usage,
                  adjusted_unbilled_total_charges         AS net_sales_billed
                FROM unbilled_revenue ur
                WHERE date_trunc('month', allocated_month) BETWEEN date_trunc('month', current_date - INTERVAL '14 month') AND date_trunc('month', current_date - INTERVAL '1 month')
                      AND adjusted_unbilled_total_charges > 0  and forecast_end_date = (select max(forecast_end_date) as forecast_end_date from unbilled_revenue)
               ),

    bills AS (select service_zone, split_part(service_zone,'_',2) as state_name, utility_account_number,
date_trunc('month', month) :: DATE AS month,billable_usage,
net_sales_billed  
  from excel_revenue_v3 er
  join config.utility_mappings um on um.am_zone = er.utility
  WHERE date_trunc('month', month) BETWEEN date_trunc('month', current_date - INTERVAL '14 month') AND date_trunc('month', current_date - INTERVAL '1 month')
                      AND net_sales_billed > 0),

     fakebills as (
         select service_zone,
                right(service_zone, 2)                                      AS state_name,
                utility_account_number,
                date_trunc('month', current_date - '1 month'::interval)::DATE as month,
                0.0                                                          as billable_usage,
                0.0                                                          as net_sales_billed
         from (
                  select service_zone, bills.utility_account_number, max(month) as maxmonth
                  from bills
                           join service_zones sz on sz.name = bills.service_zone
                           left join account_enrollment_events aee on aee.service_zone_id = sz.id and
                                                                      aee.utility_account_number = bills.utility_account_number and
                                                                      aee.action = 'drop' and
                                                                      aee.transaction_date > current_date - '2 months'::interval
                  where service_zone = 'ce_gas_mi'
                    and aee.transaction_date is null
                  group by 1, 2) foo
         where maxmonth = date_trunc('month', current_date - '2 months'::interval)::DATE),

    allbills AS (SELECT DISTINCT ON (service_zone, utility_account_number, month)
              foo.*, coalescE(base.org_type,'Legacy CRM') as org_type, base.request_id,
                 base.aggregation_id,
                 coalesce(rbc.rce,1.0) as rce
              FROM (SELECT *, 1 AS level
                        FROM bills
                        UNION
                        SELECT *, 2 AS level
                        from unbills
                        UNION
                        SELECT *, 3 as level
                        FROM fakebills) foo
              left join base on base.service_zone = foo.service_zone and base.account_number = foo.utility_account_number and base.enroll_on <= foo.month
              left join rce_by_customer rbc on rbc.service_zone = foo.service_zone and rbc.utility_account_number = foo.utility_account_number
              ORDER BY service_zone, utility_account_number, month, level ASC NULLS LAST, base.enroll_on desc nulls last)