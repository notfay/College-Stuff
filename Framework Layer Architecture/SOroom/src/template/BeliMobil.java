package template;

public class BeliMobil extends BeliKendaraan {

	public BeliMobil(String nama) {
        super(nama);
    }
	
	@Override
	public void validasiDokumen() {
		System.out.println("Mobil : " +  getNama());
	}

    @Override
    public void prosesPembayaran() {
        System.out.println("Pembayaran melalui transfer bank...");
    }

    @Override
    public void ngurusSTNKBPKB() {
        System.out.println("Mengurus STNK & BPKB mobil, estimasi 10 hari...");
    }

    @Override
    public void kirimKendaraan() {
        System.out.println("Mobil dikirim...");
    }
}
