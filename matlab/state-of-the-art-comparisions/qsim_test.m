
%% Calculate QSSIM values and store for each method
      [mqssim, ~] = qssim('Images/24jan/jpg.jpg', 'Images/24jan/jpg.jpg');
       fprintf('The QSSIM of Target  is %0.4f.\n',mqssim);
