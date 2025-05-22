select
cast((sum(p.mstotal*(ST_Area(s.subpoligono)/ ST_Area(t.poligono)))/sum((ST_Area(s.subpoligono)/ ST_Area(t.poligono)))) as NUMERIC(7,2)) as mstotal
from subarea s, potreiro t, pastagem p, medicao m 
where m.data is not NULL and s.idpotreiro=t.id and p.idsubarea=s.id and p.idmedicao=m.id and p.mstotal!=0 and p.dentrofora='F' and t.id=1
group by p.dentrofora, m.data
order by m.data;