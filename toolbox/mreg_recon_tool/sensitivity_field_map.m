function data=sensitivity_field_map(mreg_fname,ref_fname,recon_details,traj)
    
    [~, header] = loadData(mreg_fname,1);

        reference = loadReference(ref_fname);
%         recon_details.voxel_size=recon_details.voxel_size*2;
%         recon_details.recon_resolution=recon_details.recon_resolution/2;
        data = resizeReference(reference, recon_details.fov, recon_details.recon_resolution,header);
        data.raw=[];
        data.cmaps=[];
%         data.smaps      = reference.smaps;
% %         data.wmap       = reference.wmap;
%         data.anatomical = reference.anatomical;
%         data.shift      = reference.shift;
%         data.Cor_angle  = reference.Cor_angle;
%         data.ref_header = reference.header;
%         data.sensmode   = reference.sensmode;
%         data.mask       =reference.mask;

        data.trajectory =traj;
   
end