package template;

public abstract class BeliKendaraan {
	private String nama;
	
	public abstract void validasiDokumen();
	public abstract void prosesPembayaran();
	public abstract void ngurusSTNKBPKB();
	public abstract void kirimKendaraan();


	//constructor
	public BeliKendaraan() {
		super();
		this.nama = nama;
	}

	//getter setter
	public BeliKendaraan(String nama) {
		this.nama = nama;
	}

	public String getNama() {
		return nama;
	}

	public final void prosesPembelian() {
		validasiDokumen();
		prosesPembayaran();
		ngurusSTNKBPKB();
		kirimKendaraan();
	}

	
}
