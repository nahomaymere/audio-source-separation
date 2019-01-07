#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Training process module.
"""

from __future__ import print_function

import time
import os
import torch
from torch import optim
from torch.autograd import Variable

from helpers.data_feeder import data_feeder_training
from helpers.settings import debug, hyper_parameters, training_constants, \
    training_output_string, output_states_path
from modules import RNNEnc, RNNDec, FNNMasker, FNNDenoiser, TwinRNNDec, AffineTransform,CNNDenoiser,CDAE
from objectives import kullback_leibler as kl, l2_loss, sparsity_penalty, l2_reg_squared

__author__ = ['Konstantinos Drossos -- TUT', 'Stylianos Mimilakis -- Fraunhofer IDMT']
__docformat__ = 'reStructuredText'
__all__ = ['training_process']


def training_process():
    """The training process.
    """

    print('\n-- Starting training process. Debug mode: {}'.format(debug))
    print('-- Setting up modules... ', end='')
    # Masker modules
    rnn_enc = RNNEnc(hyper_parameters['reduced_dim'], hyper_parameters['context_length'], debug)
    rnn_dec = RNNDec(hyper_parameters['rnn_enc_output_dim'], debug)
    fnn = FNNMasker(
        hyper_parameters['rnn_enc_output_dim'],
        hyper_parameters['original_input_dim'],
        hyper_parameters['context_length']
    )

    # Denoiser modules
    denoiser = CNNDenoiser(hyper_parameters['original_input_dim'])
    # TwinNet regularization modules
    twin_net_rnn_dec = TwinRNNDec(hyper_parameters['rnn_enc_output_dim'], debug)
    twin_net_fnn_masker = FNNMasker(
        hyper_parameters['rnn_enc_output_dim'],
        hyper_parameters['original_input_dim'],
        hyper_parameters['context_length']
    )
    affine_transform = AffineTransform(hyper_parameters['rnn_enc_output_dim'])

    if not debug and torch.has_cudnn:
        rnn_enc = rnn_enc.cuda()
        rnn_dec = rnn_dec.cuda()
        fnn = fnn.cuda()
        denoiser = denoiser.cuda()
        twin_net_rnn_dec = twin_net_rnn_dec.cuda()
        twin_net_fnn_masker = twin_net_fnn_masker.cuda()
        affine_transform = affine_transform.cuda()

    print('done.')
    print('-- Setting up optimizes and losses... ', end='')

    # Objectives and penalties
    loss_masker = kl
    loss_denoiser = kl
    loss_twin = kl
    reg_twin = l2_loss
    reg_fnn_masker = sparsity_penalty
    reg_fnn_dec = l2_reg_squared

    # Optimizer
    optimizer = optim.Adam(
        list(rnn_enc.parameters()) +
        list(rnn_dec.parameters()) +
        list(fnn.parameters()) +
        list(denoiser.parameters()) +
        list(twin_net_rnn_dec.parameters()) +
        list(twin_net_fnn_masker.parameters()) +
        list(affine_transform.parameters()),
        lr=hyper_parameters['learning_rate']
    )

    print('done.')

    # Initializing data feeder
    epoch_it = data_feeder_training(
        window_size=hyper_parameters['window_size'],
        fft_size=hyper_parameters['fft_size'],
        hop_size=hyper_parameters['hop_size'],
        seq_length=hyper_parameters['seq_length'],
        context_length=hyper_parameters['context_length'],
        batch_size=training_constants['batch_size'],
        files_per_pass=training_constants['files_per_pass'],
        debug=debug
    )

    print('-- Training starts\n')

    # Training loop starts
    for epoch in range(training_constants['epochs']):
        epoch_l_m = []
        epoch_l_d = []
        epoch_l_tw = []
        epoch_l_twin = []

        time_start = time.time()

        # Epoch loop
        for data in epoch_it():
            v_in = Variable(torch.from_numpy(data[0]))
            v_j = Variable(torch.from_numpy(data[1]))

            if not debug and torch.has_cudnn:
                v_in = v_in.cuda()
                v_j = v_j.cuda()

            # Masker pass
            h_enc = rnn_enc(v_in)
            h_dec = rnn_dec(h_enc)
            v_j_filt_prime = fnn(h_dec, v_in)

            # TwinNet pass
            h_dec_twin = twin_net_rnn_dec(h_enc)
            v_j_filt_prime_twin = twin_net_fnn_masker(h_dec_twin, v_in)

            # Twin net regularization
            affine_output = affine_transform(h_dec)

            # Denoiser pass
            v_j_filt = denoiser(v_j_filt_prime)

            optimizer.zero_grad()

            # Calculate losses
            l_m = loss_masker(v_j_filt_prime, v_j)
            l_d = loss_denoiser(v_j_filt, v_j)
            l_tw = loss_twin(v_j_filt_prime_twin, v_j)
            l_twin = reg_twin(affine_output, h_dec_twin.detach())

            # Make MaD TwinNet objective
            loss = l_m + l_d + l_tw + (hyper_parameters['lambda_l_twin'] * l_twin) + \
                   (hyper_parameters['lambda_1'] * reg_fnn_masker(fnn.linear_layer.weight)) + \
                   (hyper_parameters['lambda_2'] * reg_fnn_dec(denoiser.fnn_dec.weight))

            # Backward pass
            loss.backward()

            # Gradient norm clipping
            torch.nn.utils.clip_grad_norm(
                list(rnn_enc.parameters()) +
                list(rnn_dec.parameters()) +
                list(fnn.parameters()) +
                list(denoiser.parameters()) +
                list(twin_net_rnn_dec.parameters()) +
                list(twin_net_fnn_masker.parameters()) +
                list(affine_transform.parameters()),
                max_norm=hyper_parameters['max_grad_norm'], norm_type=2
            )

            # Optimize
            optimizer.step()

            # Log losses
            epoch_l_m.append(l_m.data[0])
            epoch_l_d.append(l_d.data[0])
            epoch_l_tw.append(l_tw.data[0])
            epoch_l_twin.append(l_twin.data[0])

        time_end = time.time()

        # Tell us what happened
        print(training_output_string.format(
            ep=epoch,
            l_m=torch.mean(torch.FloatTensor(epoch_l_m)),
            l_d=torch.mean(torch.FloatTensor(epoch_l_d)),
            l_tw=torch.mean(torch.FloatTensor(epoch_l_tw)),
            l_twin=torch.mean(torch.FloatTensor(epoch_l_twin)),
            t=time_end - time_start
        ))

    # Kindly end and save the model
    print('\n-- Training done.')
    print('-- Saving model.. ', end='')
    torch.save(rnn_enc.state_dict(), output_states_path['rnn_enc'])
    torch.save(rnn_dec.state_dict(), output_states_path['rnn_dec'])
    torch.save(fnn.state_dict(), output_states_path['fnn'])
    torch.save(denoiser.state_dict(), output_states_path['denoiser'])
    print('done.')
    print('-- That\'s all folks!')


def main():
    currentDirectory = os.getcwd()
    workingDidrectory = os.path.dirname(currentDirectory)
    os.chdir(workingDidrectory)
    training_process()


if __name__ == '__main__':
    main()

# EOF
