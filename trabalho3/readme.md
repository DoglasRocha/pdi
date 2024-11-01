Objetivo: implemente o efeito bloom em 2 versões, com filtragem Gaussiana e box blur.
-> Para o bright-pass, não faça binarização independente de 3 canais!
-> Observe que os valores de σ aumentam bastante entre os filtros.
-> Lembre-se que a substituição não é de uma aplicação do filtro Gaussiano por uma do filtro da média; cada aplicação do filtro Gaussiano é aproximada com várias aplicações sucessivas do filtro da média!
-> Implemente (e teste) tudo o mais rápido que conseguir, não se preocupe com legibilidade, comentários, divisão em classes e funções, etc.
