% Correlation length recovery experiments

dt = 0.1;
L = 10;
conv_width = 1.0;
envelope_width = 2.0*conv_width;

tt = -L:dt:L;
ww = randn(size(tt));
gg = exp(-0.5 * (tt/conv_width).^2);
%ff = conv(gg,ww, 'same');
ff = cconv(gg,ww, length(gg));

gg2 = exp(-0.5 * (tt/envelope_width).^2);
ff2 = gg2.*ff;

plot(tt,ff)
hold on
plot(tt,ff2)
hold off

ffhat = fftshift(fft(ff));
gghat = fftshift(fft(gg));
ffhat2 = fftshift(fft(ff2));

psdf = abs(ffhat)/sum(abs(ffhat));
psdg = abs(gghat)/sum(abs(gghat));
psdf2 = abs(ffhat2)/sum(abs(ffhat2));

sqrt(var(tt, psdf))
sqrt(var(tt, psdg))
sqrt(var(tt, psdf2))

sqrt(sum(tt.^2.*psdf))

figure
plot(tt,psdf)
hold on
plot(tt,psdg, 'r')
plot(tt,psdf2, 'g')
hold off

fitdist(psdf','Normal')
fitdist(psdg','Normal')
fitdist(psdf2','Normal')

gaussEqn = 'a*exp(-((x-b)/c)^2)+d';
