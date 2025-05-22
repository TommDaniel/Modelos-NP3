select  
cast((avg(tmin)) as NUMERIC(17,2)) as tmin, 
cast((avg(tmed)) as NUMERIC(17,2)) as tmed,
cast((avg(tmax)) as NUMERIC(17,2)) as tmax,
cast((avg(umidade)) as NUMERIC(17,2)) as umidade,
cast((avg(velocidadevento)) as NUMERIC(17,2)) as velocidadevento,
cast((sum(radiacaosolar)) as NUMERIC(17,2)) as radiacaosolar,
cast((sum(chuva)) as NUMERIC(17,2)) as chuva,
cast((stddev(tmin)) as NUMERIC(17,2)) as dptmin,
cast((stddev(tmed)) as NUMERIC(17,2)) as dptmed,
cast((stddev(tmax)) as NUMERIC(17,2)) as dptmax,
cast((stddev(umidade)) as NUMERIC(17,2)) as dpumidade,
cast((stddev(velocidadevento)) as NUMERIC(17,2)) as dpvelocidadevento,
cast((stddev(radiacaosolar)) as NUMERIC(17,2)) as dpradiacaosolar,
cast((stddev(chuva)) as NUMERIC(17,2)) as dpchuva,
cast((variance(tmin)) as NUMERIC(17,2)) as vartmin,
cast((variance(tmed)) as NUMERIC(17,2)) as vartmed,
cast((variance(tmax)) as NUMERIC(17,2)) as vartmax,
cast((variance(umidade)) as NUMERIC(17,2)) as varumidade,
cast((variance(velocidadevento)) as NUMERIC(17,2)) as varvelocidadevento,
cast((variance(radiacaosolar)) as NUMERIC(17,2)) as varradiacaosolar,
cast((variance(chuva)) as NUMERIC(17,2)) as varchuva,
cast((sum(somatermica)) as NUMERIC(17,2)) as somatermica,
cast((stddev(somatermica)) as NUMERIC(17,2)) as dpsomatermica,
cast((variance(somatermica)) as NUMERIC(17,2)) as varsomatermica,
cast((sum(def)) as NUMERIC(17,2)) as def,
cast((stddev(def)) as NUMERIC(17,2)) as dpdef,
cast((variance(def)) as NUMERIC(17,2)) as vardef,
cast((sum(exc)) as NUMERIC(17,2)) as exc,
cast((stddev(exc)) as NUMERIC(17,2)) as dpexc,
cast((variance(exc)) as NUMERIC(17,2)) as varexc
from clima_evapo
WHERE data between %s and %s

--stddev
--variance