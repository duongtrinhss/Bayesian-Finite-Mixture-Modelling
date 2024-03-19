
function out = all_combination(J);

total_num = 2^J;
indicator = zeros(total_num,J);
for i = 1:J;
    temp_ones = ones( total_num/( 2^i),2^(i-1) );
    temp_zeros = zeros( total_num/(2^i),2^(i-1) );
    x_temp = [temp_ones; temp_zeros];
    indicator(:,i) = reshape(x_temp,total_num,1);
end;

out = indicator;
