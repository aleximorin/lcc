function mx = gprimage(A,x,t,clip,cm)
%
% GPRIMAGE syntax:  mx = gprimage(A,x,t,clip,cm)
% where:  A = data matrix to be plotted
%         x = position vector
%         t = time vector
%         clip = percentile above which to clip (default=98)
%         cm = colormap to be used for plot (default is grayscale)
%         mx = amplitude value used in the clipping
%
% by James Irving
% updated April, 2021

[nppt,ntr] = size(A);

if nargin==1; x=1:ntr; t=1:nppt; clip=98; cm=1-gray; end
if nargin==2; t=1:nppt; clip=98; cm=1-gray; end
if nargin==3; clip=98; cm=1-gray; end
if nargin==4; cm=1-gray; end

% determine clip (add a small value in case of zero data)
mx = percentile(abs(A(~isnan(A))),clip);
if mx==0 || isnan(mx); mx = 1e-10; end

% plot the image
imagesc(x,t,A,[-mx mx]);
colormap(cm);
set(gca,'YDir','reverse');

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function val = percentile(x,p)
    xsort = sort(x,'ascend');
    m = round((p/100)*length(x));
    val = xsort(m);
end
