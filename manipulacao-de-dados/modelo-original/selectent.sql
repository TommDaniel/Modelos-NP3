select
p.dentrofora, 
m.data,
cast((sum(p.media*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as media, 
cast((sum(p.mstotal*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as mstotal, 
cast((sum((p.msanoni/(p.msanoni+p.msoutras))*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as pcanonni
from subarea s, potreiro t, pastagem p, medicao m 
where m.data between '2013-12-20' and '2018-07-13' and s.idpotreiro=t.id and p.idsubarea=s.id and p.idmedicao=m.id and p.mstotal!=0 and t.id=1
group by p.dentrofora, m.data
order by m.data;