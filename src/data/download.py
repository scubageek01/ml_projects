import requests
import click


@click.command()
@click.argument('url')
@click.argument('filename', type=click.Path())
def download_file(url, filename):
    print('Downloading from {} to {}'.format(url, filename))
    res = requests.get(url)
    with open(filename, 'wb') as fh:
        fh.write(res.content)


if __name__ == '__main__':
    download_file()
